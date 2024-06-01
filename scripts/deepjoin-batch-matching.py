import argparse
import logging
import pandas as pd
import os
import sys
import time
import torch

from dataclasses import dataclass, field

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import HfArgumentParser, PreTrainedTokenizerBase, TrainingArguments
from typing import Dict, List, Optional, Tuple, Union

from transformers.utils import PaddingStrategy
from transformers.file_utils import cached_property, requires_backends, is_torch_tpu_available

from valentwin.algorithms.valentwin.models import RobertaForCL
from valentwin import valentine_match_pairwise
from valentwin.algorithms import DeepJoin
from valentwin.algorithms.deepjoin.train import construct_train_dataset, train_model
from valentwin.algorithms.valentwin.trainer import CLTrainer
from valentwin.embedder.text_embedder import HFTextEmbedder
from valentwin.utils.utils import convert_to_final_df

logging.basicConfig(level=logging.INFO)
logging.getLogger(__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)


# Copied from train_simcse.py
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )

    # SimCSE's arguments
    temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    pooler_type: str = field(
        default="cls",
        metadata={
            "help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last)."
        }
    )
    hard_negative_weight: float = field(
        default=0,
        metadata={
            "help": "The **logit** of weight for hard negatives (only effective if hard negatives are used)."
        }
    )
    do_mlm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use MLM auxiliary objective."
        }
    )
    mlm_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for MLM auxiliary objective (only effective if --do_mlm)."
        }
    )
    mlp_only_train: bool = field(
        default=False,
        metadata={
            "help": "Use MLP only during training"
        }
    )

    # Valentwin's arguments
    delete_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to delete the model after training."
        }
    )
    use_in_batch_instances_as_negatives: bool = field(
        default=False,
        metadata={
            "help": "Whether to use other in-batch instances as negative examples."
        }
    )
    loss_function: str = field(
        default="CrossEntropyLoss",
        metadata={
            "help": "Loss function for the model."
        }
    )
    adapter_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the adapter."
        }
    )

@dataclass
class OurTrainingArguments(TrainingArguments):
    @cached_property
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        requires_backends(self, ["torch"])
        self.distributed_state = None
        print(f"{torch.cuda.device_count()} GPUs available")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif is_torch_tpu_available():
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            self._n_gpu = 0
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
            # the default value.
            self._n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            #
            # deepspeed performs its own DDP internally, and requires the program to be started with:
            # deepspeed  ./program.py
            # rather than:
            # python -m torch.distributed.launch --nproc_per_node=2 ./program.py
            if self.deepspeed:
                from .integrations import is_deepspeed_available

                if not is_deepspeed_available():
                    raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
                import deepspeed

                deepspeed.init_distributed()
            else:
                torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", nargs="+", default=[], type=str, help="Paths to DeepJoin models to use")
    parser.add_argument("--tables_root_dirs", nargs="+", type=str, help="Path directory to the tables to be matched")
    parser.add_argument("--train_tables_root_dirs", nargs="+", default=[], type=str, help="Path directory to the training tables to be matched")
    parser.add_argument("--output_root_dirs", nargs="+", type=str, help="Path directory to save the matches")
    parser.add_argument("--column_to_text_transformations", nargs="+", type=str, default=["title-colname-stat-col"])
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--base_model_name", type=str, default=None, help="Base model name for training")
    parser.add_argument("--shuffle_rates", nargs="+", type=float, default=[0.2])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--local-rank", type=int, default=-1, help="Dummy argument for compatibility with transformers")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for i, (table_root_dir, output_root_dir) in enumerate(zip(args.tables_root_dirs, args.output_root_dirs)):
        os.makedirs(output_root_dir, exist_ok=True)

        # import tables
        fnames = sorted(fname for fname in os.listdir(table_root_dir) if fname.endswith(".csv"))
        table_names = [fname.replace(".csv", "") for fname in fnames]

        logger.info(f"Opening tables from {table_root_dir}: {table_names}")
        tables = [pd.read_csv(os.path.join(table_root_dir, fname)) for fname in fnames]
        tables_dict = {table_name: table for table_name, table in zip(table_names, tables)}

        if len(args.model_paths) > 0 and not args.train:
            for model_path in args.model_paths:
                model_name = os.path.basename(model_path)
                logger.info(f"Running DeepJoin with model: {model_name}")
                matcher = DeepJoin(model_path, column_to_text_transformation=args.column_to_text_transformations[0],
                                   all_tables=tables_dict, device=device, similarity_measure=args.similarity_measure)
                matches = valentine_match_pairwise(tables, matcher, table_names)
                matches_df = convert_to_final_df(matches, tables, table_names)
                matches_df.to_csv(os.path.join(output_root_dir, f"{model_name}-{args.similarity_measure}.csv"))

        elif args.train:  # ignore model_paths
            logger.info(f"Training DeepJoin with base model: {args.base_model_name}")
            if "simcse" in args.base_model_name:
                model = HFTextEmbedder(args.base_model_name, device=device)
            else:
                model = SentenceTransformer(args.base_model_name, device=device)
            if len(args.train_tables_root_dirs) > 0:
                logger.info(f"Opening training tables from {args.train_tables_root_dirs[i]}: {table_names}")
                train_tables = [pd.read_csv(os.path.join(args.train_tables_root_dirs[i], fname)) for fname in fnames]
                train_tables_dict = {table_name: table for table_name, table in zip(table_names, train_tables)}
            else:
                train_tables_dict = tables_dict
            for shuffle_rate in args.shuffle_rates:
                for column_to_text_transformation in args.column_to_text_transformations:
                    output_file_name = f"{os.path.basename(args.base_model_name)}-coltotext-{column_to_text_transformation}-shuffle-{shuffle_rate}-epochs-{args.num_epochs}-bs-{args.batch_size}-lr-{args.learning_rate}-wd-{args.weight_decay}-warmup-{args.warmup_steps}"
                    logger.info(f"Training DeepJoin model: {output_file_name}")

                    # Constructing dataset
                    dataset = construct_train_dataset(train_tables_dict, None, model,
                                                      column_to_text_transformation,
                                                      shuffle_rate=shuffle_rate, device=device)

                    # Training using Sentence Transformer (SBERT)
                    if isinstance(model, SentenceTransformer):
                        model = train_model(args.base_model_name, dataset, batch_size=args.batch_size,
                                            learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
                                            weight_decay=args.weight_decay, num_epochs=args.num_epochs,
                                            device=device)
                    else:  # HFTextEmbedder (training using SimCSE way)
                        parser = HfArgumentParser((ModelArguments, OurTrainingArguments))
                        model_output_dir = os.path.join(output_root_dir, "temp-"+time.time().__str__())
                        if not os.path.exists(model_output_dir):
                            os.makedirs(model_output_dir)
                        dataset_df = pd.DataFrame(dataset, columns=["source", "target"])
                        dataset_temp_file_path = os.path.join(model_output_dir, f"train.csv")
                        dataset_df.to_csv(dataset_temp_file_path, index=False)

                        args_dict = {
                            "num_train_epochs": args.num_epochs,
                            "per_device_train_batch_size": args.batch_size,
                            "learning_rate": args.learning_rate,
                            "warmup_steps": args.warmup_steps if args.warmup_steps is not None else 0,
                            "weight_decay": args.weight_decay,
                            "output_dir": model_output_dir,
                            "report_to": "none",
                            "save_strategy": "no"
                        }
                        model_args, training_args = parser.parse_dict(args_dict)

                        tokenizer = model.tokenizer
                        model_config = model.model.config

                        model = RobertaForCL.from_pretrained(
                            args.base_model_name,
                            config=model_config,
                            model_args=model_args,
                        )

                        datasets = load_dataset("csv", data_files={"train": dataset_temp_file_path}, cache_dir="./data/",
                                                delimiter=",")
                        # Prepare features
                        column_names = datasets["train"].column_names
                        # Pair datasets
                        sent0_cname = column_names[0]
                        sent1_cname = column_names[1]

                        def prepare_features(examples):
                            # padding = longest (default)
                            #   If no sentence in the batch exceed the max length, then use
                            #   the max sentence length in the batch, otherwise use the
                            #   max sentence length in the argument and truncate those that
                            #   exceed the max length.
                            # padding = max_length (when pad_to_max_length, for pressure test)
                            #   All sentences are padded/truncated to data_args.max_seq_length.
                            total = len(examples[sent0_cname])

                            # Avoid "None" fields
                            for idx in range(total):
                                if examples[sent0_cname][idx] is None:
                                    examples[sent0_cname][idx] = " "
                                if examples[sent1_cname][idx] is None:
                                    examples[sent1_cname][idx] = " "

                            sentences = examples[sent0_cname] + examples[sent1_cname]

                            sent_features = tokenizer(
                                sentences,
                                max_length=512,
                                truncation=True,
                                padding=False,
                            )

                            features = {}
                            for key in sent_features:
                                features[key] = [[sent_features[key][i], sent_features[key][i + total]] for i in
                                                 range(total)]

                            return features

                        train_dataset = datasets["train"].map(
                            prepare_features,
                            batched=True,
                            remove_columns=column_names,
                            load_from_cache_file=True,
                        )

                        @dataclass
                        class OurDataCollatorWithPadding:

                            tokenizer: PreTrainedTokenizerBase
                            padding: Union[bool, str, PaddingStrategy] = True
                            max_length: Optional[int] = None
                            pad_to_multiple_of: Optional[int] = None

                            def __call__(self,
                                         features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> \
                            Dict[
                                str, torch.Tensor]:
                                special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids',
                                                'mlm_labels']
                                valentwin_keys = ['column_names', 'table_names']
                                bs = len(features)
                                if bs > 0:
                                    num_sent = len(features[0]['input_ids'])
                                else:
                                    return
                                flat_features = []
                                for feature in features:
                                    for i in range(num_sent):
                                        flat_features.append(
                                            {k: feature[k][i] if k in special_keys else feature[k] for k in feature if
                                             k not in valentwin_keys})

                                batch = self.tokenizer.pad(
                                    flat_features,
                                    padding=self.padding,
                                    max_length=self.max_length,
                                    pad_to_multiple_of=self.pad_to_multiple_of,
                                    return_tensors="pt",
                                )

                                batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs,
                                                                                                                    num_sent,
                                                                                                                    -1)[
                                                                                                      :, 0]
                                         for k in batch}

                                if "label" in batch:
                                    batch["labels"] = batch["label"]
                                    del batch["label"]
                                if "label_ids" in batch:
                                    batch["labels"] = batch["label_ids"]
                                    del batch["label_ids"]

                                if "column_names" in features[0]:
                                    batch["column_names"] = [feature["column_names"] for feature in features]
                                if "table_names" in features[0]:
                                    batch["table_names"] = [feature["table_names"] for feature in features]

                                return batch


                        data_collator = OurDataCollatorWithPadding(tokenizer)

                        trainer = CLTrainer(
                            model=model,
                            args=training_args,
                            train_dataset=train_dataset,
                            tokenizer=tokenizer,
                            data_collator=data_collator,
                        )

                        train_result = trainer.train()
                        tokenizer = trainer.tokenizer
                        model = trainer.model
                        model = HFTextEmbedder(model=model, tokenizer=tokenizer, device=device, converted_to_hf=False)
                        # Delete the temporary folder
                        os.system(f"rm -rf {model_output_dir}")
                        logger.info("Training completed.")

                    # Matching
                    logger.info("Matching tables with the trained model.")
                    matcher = DeepJoin(model=model, column_to_text_transformation=column_to_text_transformation,
                                       all_tables=tables_dict)
                    matches = valentine_match_pairwise(tables, matcher, table_names)
                    matches_df = convert_to_final_df(matches, tables, table_names)
                    matches_df.to_csv(os.path.join(output_root_dir, f"deepjoin-{output_file_name}.csv"))

        else:
            raise ValueError("No model paths provided and not training a model.")


    logger.info("Done!")
    sys.exit()


if __name__ == "__main__":
    main()
