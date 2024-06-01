import argparse
import logging
import pandas as pd
import os
import torch

from tqdm.auto import tqdm

from valentwin import valentine_match_pairwise
from valentwin.algorithms import ValenTwin
from valentwin.embedder.text_embedder import HFTextEmbedder
from valentwin.utils.utils import convert_to_final_df

logging.basicConfig(level=logging.INFO)
logging.getLogger(__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_names_or_paths", nargs="+", default=[], type=str, help="Model IDs or paths to HF model")
    parser.add_argument("--pooling", type=str, default="cls", help="Pooling strategy for the model")
    parser.add_argument("--inference_batch_size", type=int, default=128, help="Batch size for inference")
    parser.add_argument("--column_name_weights", nargs="+", default=[], type=float, help="Weights for column names")
    parser.add_argument("--measures", nargs="+", default=[], type=str, help="Measures to use for matching")
    parser.add_argument("--column_name_measures", nargs="+", default=[], type=str, help="Measures to use for column names")
    parser.add_argument("--holistic", action="store_true", help="Whether to use holistic matching")
    parser.add_argument("--tables_root_dir", type=str, help="Path directory to the tables to be matched")
    parser.add_argument("--output_root_dir", type=str, help="Path directory to save the matches")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use for the model")

    args = parser.parse_args()

    os.makedirs(args.output_root_dir, exist_ok=True)

    # import tables
    fnames = sorted(fname for fname in os.listdir(args.tables_root_dir) if fname.endswith(".csv"))
    table_names = [fname.replace(".csv", "") for fname in fnames]

    logger.info(f"Opening tables from {args.tables_root_dir}: {table_names}")
    tables = [pd.read_csv(os.path.join(args.tables_root_dir, fname)) for fname in fnames]

    for i, pretrained_model_name_or_path in tqdm(enumerate(args.pretrained_model_names_or_paths)):
        logger.info(f"Loading model {pretrained_model_name_or_path}")
        text_embedder = HFTextEmbedder(pretrained_model_name_or_path, device=args.device, use_auth_token=True,
                                       inference_batch_size=args.inference_batch_size,
                                       pooling=args.pooling, use_cache=True)

        model_name = pretrained_model_name_or_path.split("/")[-1]
        logger.info(f"Matching tables with {model_name}")

        default_measures = {"cos": "cosine_similarity", "emd": "earth_movers_distance", "euc": "euclidean_distance"}
        measures = {measure: default_measures.get(measure) for measure in args.measures} if args.measures else default_measures
        column_name_measures = {measure: default_measures.get(measure) for measure in args.column_name_measures} if len(args.column_name_measures) > 0 else {None: None}

        column_name_weights = args.column_name_weights if args.column_name_weights else [0.0, 0.1, 0.2, 0.3]
        for measure_key, measure_name in measures.items():
            for column_name_measure_key, column_name_measure_name in column_name_measures.items():
                for weight in column_name_weights:
                    matcher = ValenTwin(text_embedder, column_name_weight=weight,
                                        measure=measure_name, column_name_measure=column_name_measure_name)

                    matches = valentine_match_pairwise(tables, matcher, table_names, holistic=args.holistic)
                    matches_df = convert_to_final_df(matches, tables, table_names)
                    fname = f"{model_name}-{measure_key}-cnw-{str(weight).replace('.', '')}{'-'+str(column_name_measure_key) if column_name_measure_key is not None else ''}.csv"
                    matches_df.to_csv(os.path.join(args.output_root_dir, fname))

    logger.info("Done!")


if __name__ == "__main__":
    main()
