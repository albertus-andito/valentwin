import argparse
import logging
import pandas as pd
import os
import torch

from tqdm.auto import tqdm

from valentwin.embedder.text_embedder import HFTextEmbedder
from valentwin.embedder.turl_embedder import TurlEmbedder
from valentwin import valentine_match_pairwise
from valentwin.algorithms import ALITE
from valentwin.utils.utils import convert_to_final_df

logging.basicConfig(level=logging.INFO)
logging.getLogger(__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_names_or_paths", nargs="+", default=[], type=str, help="Model IDs or paths to HF model")
    parser.add_argument("--is_table_embedder", action="store_true", default=False, help="Whether the model is a table embedder, e.g. TURL")
    parser.add_argument("--table_embedder_data_dir", type=str, default=None, help="Path to the data directory for the table embedder")
    parser.add_argument("--table_embedder_config_path", type=str, default=None, help="Path to the config file for the table embedder")
    parser.add_argument("--pre_computed_embeddings_file_path", type=str, default=None, help="Path to the pre-computed embeddings JSON file")
    parser.add_argument("--column_names_map_file_path", type=str, default=None, help="Path to the column names map CSV file")

    parser.add_argument("--tables_root_dirs", nargs="+", type=str, help="Paths directory to the tables to be matched")
    parser.add_argument("--output_root_dirs", nargs="+", type=str, help="Paths directory to save the matches")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use for the model")
    parser.add_argument("--clustering_distance_metric", type=str, default="l2", help="Distance metric to use for clustering")

    args = parser.parse_args()

    for tables_root_dir, output_root_dir in zip(args.tables_root_dirs, args.output_root_dirs):
        os.makedirs(output_root_dir, exist_ok=True)

        # import tables
        fnames = sorted(fname for fname in os.listdir(tables_root_dir) if fname.endswith(".csv"))
        table_names = [fname.replace(".csv", "") for fname in fnames]

        logger.info(f"Opening tables from {args.tables_root_dir}: {table_names}")
        tables = [pd.read_csv(os.path.join(args.tables_root_dir, fname)) for fname in fnames]

        for pretrained_model_name_or_path in tqdm(args.pretrained_model_names_or_paths):
            logger.info(f"Loading model {pretrained_model_name_or_path}")
            if not args.is_table_embedder:
                text_embedder = HFTextEmbedder(pretrained_model_name_or_path, device=args.device, use_auth_token=True,
                                               use_cache=True)
                table_embedder = None
            else:
                text_embedder = None
                table_embedder = TurlEmbedder(data_dir=args.table_embedder_data_dir,
                                              config_name=args.table_embedder_config_path,
                                              model_name_or_path=pretrained_model_name_or_path,
                                              device=args.device)

            model_name = pretrained_model_name_or_path.split("/")[-1]
            logger.info(f"Matching tables with {model_name}")

            matcher = ALITE(text_embedder, table_embedder, clustering_distance_metric=args.clustering_distance_metric)
            matches = valentine_match_pairwise(tables, matcher, table_names)
            matches_df = convert_to_final_df(matches, tables, table_names)
            matches_df.to_csv(os.path.join(args.output_root_dir, f"alite-{model_name}-{args.clustering_distance_metric}.csv"))

        if args.pre_computed_embeddings_file_path is not None:
            logger.info(f"Matching tables with pre-computed embeddings from file:{args.pre_computed_embeddings_file_path}")

            matcher = ALITE(None, None,
                            pre_computed_embeddings_file_path=args.pre_computed_embeddings_file_path,
                            column_names_map_file_path=args.column_names_map_file_path,
                            clustering_distance_metric=args.clustering_distance_metric)
            matches = valentine_match_pairwise(tables, matcher, table_names)
            matches_df = convert_to_final_df(matches, tables, table_names)

            if "/bert" in args.pre_computed_embeddings_file_path:
                model_name = "precomputed-bert"
            elif "/turl" in args.pre_computed_embeddings_file_path:
                model_name = "precomputed-turl"
            else:
                model_name = "precomputed-fasttext"

            matches_df.to_csv(os.path.join(args.output_root_dir, f"alite-{model_name}-{args.clustering_distance_metric}.csv"))


if __name__ == "__main__":
    main()