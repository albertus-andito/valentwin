import argparse
import logging
import os
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from typing import Dict, Any

from valentwin.metrics import all_metrics
from valentwin.utils.utils import annotate_tp_fp, ColumnTypes

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("Logging started")


def read_matches_df(filepath):
    # read matches_df
    matches_df = pd.read_csv(filepath, index_col=0)
    matches = {}
    for i, row in matches_df.iterrows():
        if "type" in row:
            matches[((row["table_a"], row["column_a"]), (row["table_b"], row["column_b"]), row["type"])] = row[
                "similarity"]
        else:
            matches[((row["table_a"], row["column_a"]), (row["table_b"], row["column_b"]))] = row["similarity"]
    return matches


def process_file(file, input_dir_path, all_table_columns,
                 only_textual_columns, split_by_column_types, textual_ground_truth_pairs,
                 numerical_ground_truth_pairs, ground_truth_pairs, correct_col_name, do_annotate_tp_fp):
    print(file)
    matches = read_matches_df(os.path.join(input_dir_path, file))

    if only_textual_columns:
        to_delete = []
        for k in matches.keys():
            if k[0] not in all_table_columns or k[1] not in all_table_columns:
                to_delete.append(k)
        for col in to_delete:
            del matches[col]

    complete_metrics = {}

    if split_by_column_types and len(next(iter(matches))) == 3:
        textual_matches = {k: v for k, v in matches.items() if k[2] == "textual" or k[2] == "mixed"}
        numerical_matches = {k: v for k, v in matches.items() if k[2] == "numerical"}

        textual_metrics = all_metrics(textual_matches, textual_ground_truth_pairs, one_to_one=False, one_to_one_intra_table=True)
        numerical_metrics = all_metrics(numerical_matches, numerical_ground_truth_pairs, one_to_one=False, one_to_one_intra_table=True)

        textual_metrics = {f"textual_{k}": v for k, v in textual_metrics.items()}
        numerical_metrics = {f"numerical_{k}": v for k, v in numerical_metrics.items()}

        complete_metrics = {**textual_metrics, **numerical_metrics}
    if "hybrid" not in file:
        metrics = all_metrics(matches, ground_truth_pairs, one_to_one=False, one_to_one_intra_table=True)
        complete_metrics = {**complete_metrics, **metrics}

    if do_annotate_tp_fp:
        # logger.info("Annotating TP and FP")
        matches_df = pd.read_csv(os.path.join(input_dir_path, file), index_col=0)
        if correct_col_name not in matches_df.columns:
            matches_df = annotate_tp_fp(matches_df, ground_truth_pairs, correct_col_name=correct_col_name)
            matches_df.to_csv(os.path.join(input_dir_path, file))
    return file.replace(".csv", ""), complete_metrics


def main(args_dict: Dict[str, Any] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir_path", type=str, help="Path to the directory containing the matching result files")
    parser.add_argument("--ground_truth_file_path", type=str, help="Path to the ground truth file")
    parser.add_argument("--output_file_path", type=str, help="Path to the output file")
    parser.add_argument("--do_annotate_tp_fp", action="store_true", help="Whether to annotate TP and FP in the matching results")
    parser.add_argument("--correct_col_name", type=str, default="correct", help="Name of the column to store TP and FP")
    parser.add_argument("--only_textual_columns", action="store_true", help="Whether to consider only textual columns")
    parser.add_argument("--split_by_column_types", action="store_true", help="Whether to split the metrics by column types")
    parser.add_argument("--overwrite_computed_metrics", action="store_true", help="Whether to overwrite the computed metrics")
    parser.add_argument("--parallel_workers", type=int, default=1, help="Number of parallel workers")
    args = parser.parse_args()

    logger.info("Reading ground truth file")
    ground_truth_df = pd.read_csv(args.ground_truth_file_path, index_col=0)
    if args.only_textual_columns:
        ground_truth_pairs = [((row["source_table"], str(row["source_column"]).strip()),
                               (row["target_table"], str(row["target_column"]).strip()))
                              for i, row in ground_truth_df.iterrows() if row["type"] == ColumnTypes.TEXTUAL.value]
    else:
        ground_truth_pairs = [((row["source_table"], str(row["source_column"]).strip()),
                               (row["target_table"], str(row["target_column"]).strip()))
                              for i, row in ground_truth_df.iterrows()]
    if "type" in ground_truth_df.columns:
        textual_ground_truth_pairs = [((row["source_table"], str(row["source_column"]).strip()),
                                       (row["target_table"], str(row["target_column"]).strip()))
                                      for i, row in ground_truth_df.iterrows() if row["type"] == ColumnTypes.TEXTUAL.value or row["type"] == ColumnTypes.MIXED.value] # note that we're including mixed type as textual here
        numerical_ground_truth_pairs = [((row["source_table"], str(row["source_column"]).strip()),
                                         (row["target_table"], str(row["target_column"]).strip()))
                                        for i, row in ground_truth_df.iterrows() if row["type"] == ColumnTypes.NUMERICAL.value]
    else:
        textual_ground_truth_pairs = ground_truth_pairs
        numerical_ground_truth_pairs = ground_truth_pairs

    all_table_columns = set([col for pair in ground_truth_pairs for col in pair])

    if not args.overwrite_computed_metrics:
        metrics_df = pd.read_csv(args.output_file_path, index_col=0)

    logger.info("Calculating metrics")
    files = [f for f in sorted(os.listdir(args.input_dir_path)) if f.endswith(".csv")]
    if not args.overwrite_computed_metrics:
        files = [f for f in files if f.replace(".csv", "") not in metrics_df.index]
    final_metrics = {}

    if args.parallel_workers > 1:
        with tqdm(total=len(files)) as pbar:
            max_workers = None if args.parallel_workers == -1 else args.parallel_workers
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_file, f, args.input_dir_path, all_table_columns,
                                           args.only_textual_columns, args.split_by_column_types, textual_ground_truth_pairs,
                                           numerical_ground_truth_pairs, ground_truth_pairs, args.correct_col_name,
                                           args.do_annotate_tp_fp) for f in files]

                for future in as_completed(futures):
                    file, metrics = future.result()
                    final_metrics[file] = metrics
                    pbar.update(1)
    else:
        for file in tqdm(files):
            file, metrics = process_file(file, args.input_dir_path, all_table_columns,
                                         args.only_textual_columns, args.split_by_column_types, textual_ground_truth_pairs,
                                         numerical_ground_truth_pairs, ground_truth_pairs, args.correct_col_name,
                                         args.do_annotate_tp_fp)
            final_metrics[file] = metrics

    if args.overwrite_computed_metrics:
        final_metrics_df = pd.DataFrame.from_dict(final_metrics, orient="index")
    else:
        final_metrics_df = pd.concat([metrics_df, pd.DataFrame.from_dict(final_metrics, orient="index")])

    os.makedirs(os.path.dirname(args.output_file_path), exist_ok=True)
    final_metrics_df.to_csv(args.output_file_path)

    logger.info("Done")


if __name__ == "__main__":
    main()
