import argparse
import logging
import pandas as pd
import os
import sys

from valentwin import valentine_match_pairwise
from valentwin.algorithms import Cupid, JaccardDistanceMatcher, DistributionBased, SimilarityFlooding, Coma
from valentwin.utils.utils import convert_to_final_df

logging.basicConfig(level=logging.INFO)
logging.getLogger(__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithms", nargs="+", default=[], type=str, help="Matching algorithms to use")
    parser.add_argument("--tables_root_dirs", nargs="+", type=str, help="Path directory to the tables to be matched")
    parser.add_argument("--output_root_dirs", nargs="+", type=str, help="Path directory to save the matches")
    parser.add_argument("--parallelize", action="store_true", help="Whether to parallelize the matching process")

    args = parser.parse_args()

    for table_root_dir, output_root_dir in zip(args.tables_root_dirs, args.output_root_dirs):
        os.makedirs(output_root_dir, exist_ok=True)

        # import tables
        fnames = sorted(fname for fname in os.listdir(table_root_dir) if fname.endswith(".csv"))
        table_names = [fname.replace(".csv", "") for fname in fnames]

        logger.info(f"Opening tables from {table_root_dir}: {table_names}")
        tables = [pd.read_csv(os.path.join(table_root_dir, fname)) for fname in fnames]

        for algorithm in args.algorithms:
            logger.info(f"Running {algorithm} algorithm.")
            if algorithm == "cupid":
                matcher = Cupid()
            elif algorithm == "jaccard":
                matcher = JaccardDistanceMatcher()
            elif algorithm == "distribution_based":
                matcher = DistributionBased()
            elif algorithm == "similarity_flooding":
                matcher = SimilarityFlooding()
            elif algorithm == "coma_schema":
                matcher = Coma(use_instances=False)
            elif algorithm == "coma_schema_instance":
                matcher = Coma(use_instances=True)
            else:
                raise ValueError(f"Unknown algorithm {algorithm}")
            matches = valentine_match_pairwise(tables, matcher, table_names, parallelize=args.parallelize)
            matches_df = convert_to_final_df(matches, tables, table_names)
            matches_df.to_csv(os.path.join(output_root_dir, f"{algorithm}.csv"))

    logger.info("Done!")
    sys.exit()



if __name__ == "__main__":
    main()
