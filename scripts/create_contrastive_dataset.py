import argparse
import logging
import os
import torch

from tqdm.auto import tqdm

from valentwin.algorithms.valentwin.contrastive_dataset_generator import ContrastiveDatasetGenerator
from valentwin.embedder.text_embedder import HFTextEmbedder

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("Logging started")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir_paths", nargs="+", type=str, help="Paths to the directory containing the table files")
    parser.add_argument("--output_dir_paths", nargs="+", type=str, help="Path to the output directory")
    parser.add_argument("--hard_neg_size", type=int, default=10, help="Number of hard negative samples to generate")
    parser.add_argument("--with_col_table_names", action="store_true", help="Whether to include column and table names in the generated samples")
    parser.add_argument("--use_selective_negatives", action="store_true", help="Whether to use selective negatives")
    parser.add_argument("--use_selective_positives", action="store_true", help="Whether to use selective positives")
    parser.add_argument("--pretrained_model_name_or_path", type=str, help="Path to the pretrained model")
    parser.add_argument("--pooling", type=str, default="cls", help="Pooling method")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use")
    args = parser.parse_args()

    logger.info("Starting constructing contrastive dataset")
    generator = ContrastiveDatasetGenerator()
    if args.use_selective_negatives:
        text_embedder = HFTextEmbedder(args.pretrained_model_name_or_path, pooling="cls", use_cache=True,
                                       use_auth_token=True, device=args.device)
    else:
        text_embedder = None

    for input_dir_path, output_dir_path in tqdm(zip(args.input_dir_paths, args.output_dir_paths),
                                                total=len(args.input_dir_paths)):
        filepaths = [os.path.join(input_dir_path, fname) for fname in os.listdir(input_dir_path)]
        generator.generate(filepaths, output_dir_path, supervised=False, train_val_test_proportion=[0.7, 0.1, 0.2],
                           hard_neg_size=args.hard_neg_size, with_col_table_names=args.with_col_table_names,
                           use_selective_negatives=args.use_selective_negatives,
                           use_selective_positives=args.use_selective_positives,
                           text_embedder=text_embedder)
    logger.info("Done!")


if __name__ == "__main__":
    main()