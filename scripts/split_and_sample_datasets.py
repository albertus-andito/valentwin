import argparse
import logging

from tqdm.auto import tqdm

from valentwin.algorithms.valentwin.dataset_utils import split_and_sample_datasets

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("Logging started")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir_paths", nargs="+", type=str, help="Paths to the directory containing the table files")
    parser.add_argument("--sample_sizes", nargs="+", type=int, help="List of sample sizes")
    parser.add_argument("--sample_dataset_dir_paths", nargs="+", type=str, help="Paths to the output directory")
    parser.add_argument("--split_ratio", nargs="+", type=float, help="List of split ratios")
    parser.add_argument("--include_all_samples", action="store_true", help="Whether to include all samples")
    parser.add_argument("--drop_duplicates", action="store_true", help="Whether to drop duplicates")
    parser.add_argument("--first_col_index", action="store_true", help="Whether the first column is an index")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    logger.info("Starting splitting and sampling datasets")
    for dataset_dir_path, sample_dataset_dir_path in tqdm(zip(args.dataset_dir_paths, args.sample_dataset_dir_paths),
                                                          total=len(args.dataset_dir_paths)) :
        split_and_sample_datasets(dataset_dir_path, args.sample_sizes, sample_dataset_dir_path, args.split_ratio,
                                  include_all_samples=args.include_all_samples, drop_duplicates=args.drop_duplicates,
                                  seed=args.seed)
    logger.info("Done!")


if __name__ == "__main__":
    main()