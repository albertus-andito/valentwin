import logging
import pandas as pd
import os

from typing import List


logger = logging.getLogger(__name__)


def sample_with_at_least_n_value_per_column(df: pd.DataFrame, sample_size: int, n: int = 1, seed: int = 42,
                                            drop_duplicates: bool = True):
    df = df.copy()

    if drop_duplicates:
        df = df.drop_duplicates()
    selected_indices = set()
    for column in df.columns:
        non_empty_rows = df[df[column].notna()]
        if not non_empty_rows.empty:
            selected_index = non_empty_rows.sample(n, random_state=seed).index
            selected_indices.add(selected_index.item())

    if sample_size > len(df):
        sample_size = len(df.drop(list(selected_indices)))

    # Before sampling additional rows
    if len(selected_indices) < sample_size:
        # Calculate the number of additional samples needed
        additional_samples_needed = sample_size - len(selected_indices)

        # Check if there are enough rows left to sample from
        remaining_rows = df.drop(list(selected_indices))
        if len(remaining_rows) >= additional_samples_needed:
            # If enough rows are available, proceed to sample additional rows
            additional_samples = remaining_rows.sample(n=additional_samples_needed, random_state=seed)
        else:
            additional_samples = remaining_rows.sample(n=len(remaining_rows), random_state=seed)
    else:
        # If no additional samples are needed, proceed without sampling more rows
        additional_samples = pd.DataFrame()

    sample_df = pd.concat([df.loc[list(selected_indices)], additional_samples])
    return sample_df


def create_samples(a_df: pd.DataFrame, sample_sizes: List[int], sample_dataset_dir_path: str, seed: int,
                   set_type: str, fname: str, drop_duplicates: bool = True, include_all_sample: bool = True):
    prev_sample_df = pd.DataFrame()
    prev_sample_size = 0
    for sample_size in sorted(sample_sizes):  # next sample size should include sample from the previous size
        sample_dir = os.path.join(sample_dataset_dir_path, f"{str(sample_size)}-{set_type}")
        os.makedirs(sample_dir, exist_ok=True)

        size_to_sample = sample_size - prev_sample_size
        sample_df = sample_with_at_least_n_value_per_column(a_df, max(0, size_to_sample), n=1, seed=seed,
                                                            drop_duplicates=drop_duplicates)
        current_df = pd.concat([prev_sample_df, sample_df])
        current_df.to_csv(os.path.join(sample_dir, f"{fname}"), index=False)
        prev_sample_df = current_df
        prev_sample_size = sample_size
        a_df = a_df.drop(sample_df.index)

    if include_all_sample:
        sample_dir = os.path.join(sample_dataset_dir_path, f"all-{set_type}")
        os.makedirs(sample_dir, exist_ok=True)
        current_df = pd.concat([prev_sample_df, a_df])
        current_df.to_csv(os.path.join(sample_dir, f"{fname}"), index=False)


def split_and_sample_datasets(dataset_dir_path: str, sample_sizes: List[int], sample_dataset_dir_path: str,
                              split_ratio: List[float] = None,
                              include_all_samples: bool = True, drop_duplicates: bool = True, seed: int = 42,
                              first_col_index: bool = False):
    logging.info(f"Splitting and sampling datasets in {dataset_dir_path}")
    if split_ratio is None:
        split_ratio = [0.4, 0.2, 0.4]  # train, validation, test
    for fname in os.listdir(dataset_dir_path):
        if not fname.endswith(".csv"):
            continue
        if first_col_index:
            df = pd.read_csv(os.path.join(dataset_dir_path, fname), index_col=0)
        else:
            df = pd.read_csv(os.path.join(dataset_dir_path, fname))
        if drop_duplicates:
            df = df.drop_duplicates()
        train_df = df.sample(frac=split_ratio[0], random_state=seed)
        if len(split_ratio) == 3:
            validation_df = df.drop(train_df.index).sample(frac=split_ratio[1] / (split_ratio[1] + split_ratio[2]),
                                                           random_state=seed)
            test_df = df.drop(train_df.index).drop(validation_df.index).sample(frac=1, random_state=seed)
        else:  # 2 splits (train and test)
            test_df = df.drop(train_df.index).sample(frac=1, random_state=seed)

        create_samples(train_df, sample_sizes, sample_dataset_dir_path, seed, "train", fname, drop_duplicates, include_all_samples)
        if len(split_ratio) == 3:
            create_samples(validation_df, sample_sizes, sample_dataset_dir_path, seed, "validation", fname, drop_duplicates, include_all_samples)
        create_samples(test_df, sample_sizes, sample_dataset_dir_path, seed, "test", fname, drop_duplicates, include_all_samples)
