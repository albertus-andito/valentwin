import argparse
import logging
import numpy as np
import pandas as pd
import os
import random
import sys
import torch

from pathlib import Path

from valentwin import valentine_match_pairwise
from valentwin.algorithms.starmie.dataset import PretrainTableDataset
from valentwin.algorithms.starmie.pretrain import train
from valentwin.algorithms import Starmie
from valentwin.utils.utils import convert_to_final_df

logging.basicConfig(level=logging.INFO)
logging.getLogger(__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", nargs="+", default=[], type=str, help="Starmie models to use")
    parser.add_argument("--tables_root_dirs", nargs="+", type=str, help="Path directory to the tables to be matched")
    parser.add_argument("--train_tables_root_dirs", nargs="+", type=str, default=[], help="Path directory to the tables to be used for training")
    parser.add_argument("--output_root_dirs", nargs="+", type=str, help="Path directory to save the matches")
    parser.add_argument("--fine_tune", dest="fine_tune", action="store_true")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--lm", type=str, default='roberta-base')
    parser.add_argument("--projector", type=int, default=768)
    parser.add_argument("--augment_op", type=str, default='drop_col,sample_row')
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    # single-column mode without table context
    parser.add_argument("--single_column", dest="single_column", action="store_true")
    # row / column-ordered for preprocessing
    parser.add_argument("--table_order", type=str, default='column')
    # for sampling
    parser.add_argument("--sample_meth", type=str, default='head')

    args = parser.parse_args()

    for i, (table_root_dir, output_root_dir) in enumerate(zip(args.tables_root_dirs, args.output_root_dirs)):
        os.makedirs(output_root_dir, exist_ok=True)

        # import tables
        fnames = sorted(fname for fname in os.listdir(table_root_dir) if fname.endswith(".csv"))
        table_names = [fname.replace(".csv", "") for fname in fnames]

        logger.info(f"Opening tables from {table_root_dir}: {table_names}")
        tables = [pd.read_csv(os.path.join(table_root_dir, fname)) for fname in fnames]

        if len(args.train_tables_root_dirs) > 0:
            train_table_root_dir = args.train_tables_root_dirs[i]
        else:
            train_table_root_dir = table_root_dir

        if len(args.model_paths) > 0:
            if os.path.isdir(args.model_paths[0]):
                args.model_paths = [os.path.join(args.model_paths[0], model) for model in
                                    os.listdir(args.model_paths[0])]
            for model_path in args.model_paths:
                model_name = Path(model_path).stem
                if args.fine_tune:
                    logger.info(f"Fine-tuning Starmie with model: {model_name}")
                    ckpt = torch.load(model_path,
                                      map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                    # hyperparameters mostly follow the pretraining
                    hp = ckpt['hp']
                    hp.task = train_table_root_dir
                    hp.save_model = False
                    logger.info(f"Fine-tuning with hyperparameters: {hp}")
                    trainset = PretrainTableDataset.from_hp(train_table_root_dir, hp)
                    model = train(trainset, hp, state_dict=ckpt['model'])
                    matcher = Starmie(model_path=None, model=model, hp=hp, use_cache=True, similarity_measure=args.similarity_measure)
                    model_name += "-ft"
                else:

                    logger.info(f"Running Starmie with model: {model_name}")
                    matcher = Starmie(model_path, similarity_measure=args.similarity_measure)
                matches = valentine_match_pairwise(tables, matcher, table_names)
                matches_df = convert_to_final_df(matches, tables, table_names)
                matches_df.to_csv(os.path.join(output_root_dir, f"starmie-{model_name}-{args.similarity_measure}.csv"))
        else:
            logger.info("No models provided, training Starmie on the tables")
            if args.augment_op == "all":
                augment_ops = ["drop_col", "sample_row", "sample_row_ordered", "shuffle_col", "drop_cell",
                               "sample_cells", "replace_cells", "drop_head_cells", "drop_num_cells", "swap_cells",
                               "drop_num_col", "drop_nan_col"]
            else:
                augment_ops = [args.augment_op]
            if args.sample_meth == "all":
                sample_meths = ["head", "alphaHead", "random", "constant", "frequent", "tfidf_token", "tfidf_entity",
                                "tfidf_row"]
            else:
                sample_meths = [args.sample_meth]
            if args.table_order == "all":
                table_orders = ["column", "row"]
            else:
                table_orders = [args.table_order]
            # Train model and run Starmie
            hp = parser.parse_args()
            hp.task = train_table_root_dir
            hp.save_model = False
            for augment_op in augment_ops:
                for sample_meth in sample_meths:
                    for table_order in table_orders:
                        hp.table_order = table_order
                        hp.augment_op = augment_op
                        hp.sample_meth = sample_meth

                        # set seeds
                        seed = hp.run_id
                        random.seed(seed)
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(seed)

                        path = train_table_root_dir
                        trainset = PretrainTableDataset.from_hp(path, hp)

                        model = train(trainset, hp)
                        model_name = str(hp.augment_op)+'_'+str(hp.sample_meth)+'_'+str(hp.table_order)+'_'+str(hp.run_id)
                        matcher = Starmie(model_path=None, model=model, hp=hp, use_cache=True)
                        matches = valentine_match_pairwise(tables, matcher, table_names)
                        matches_df = convert_to_final_df(matches, tables, table_names)
                        matches_df.to_csv(os.path.join(output_root_dir, f"starmie-{model_name}.csv"))

    logger.info("Done!")
    sys.exit()


if __name__ == "__main__":
    main()
