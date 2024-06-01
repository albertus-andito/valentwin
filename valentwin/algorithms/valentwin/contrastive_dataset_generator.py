from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from typing import Dict, List

import logging
import numpy as np
import os
import pandas as pd
import random

from valentwin.algorithms.valentwin.utils import cos_sim
from valentwin.embedder.text_embedder import HFTextEmbedder
from valentwin.utils.utils import convert_str_to_list


class ContrastiveDatasetGenerator:
    """
    Generates contrastive datasets from tables.
    """
    SENT0_COL = "sent0"
    SENT1_COL = "sent1"
    HARD_NEG_COL = "hard_neg"
    POS_COL_NAME = "pos_col_name"
    NEG_COL_NAME = "neg_col_name"
    TABLE_NAME = "table_name"

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(self.seed)
        self.logger = logging.getLogger(__name__)

    def generate(self, input_file_paths: List[str], output_dir: str, supervised: bool = False,
                 expand_list: bool = False, remove_duplicates: bool = True,
                 train_val_test_proportion: List[float] = None, hard_neg_size: int = 10,
                 with_col_table_names: bool = True,
                 use_selective_negatives: bool = True, use_selective_positives: bool = False,
                 text_embedder: HFTextEmbedder = None):
        tables = {Path(input_file_path).stem: pd.read_csv(input_file_path) for input_file_path in input_file_paths}

        if supervised:
            raise NotImplementedError("Supervised contrastive dataset generation is not implemented yet")
        else:
            train_df, val_df, test_df = self.generate_from_unsupervised(tables,
                                                                        expand_list=expand_list,
                                                                        remove_duplicates=remove_duplicates,
                                                                        train_val_test_proportion=train_val_test_proportion,
                                                                        hard_neg_size=hard_neg_size,
                                                                        return_col_table_names=with_col_table_names,
                                                                        use_selective_negatives=use_selective_negatives,
                                                                        use_selective_positives=use_selective_positives,
                                                                        text_embedder=text_embedder)

        os.makedirs(output_dir, exist_ok=True)

        train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)


    def generate_from_unsupervised(self, tables: Dict[str, pd.DataFrame],
                                   expand_list: bool = False, remove_duplicates: bool = True,
                                   train_val_test_proportion: List[float] = None, hard_neg_size: int = 10,
                                   return_col_table_names: bool = False,
                                   use_selective_negatives: bool = False,
                                   use_selective_positives: bool = False, positive_pairs_size: int = 25,
                                   text_embedder: HFTextEmbedder = None):
        if train_val_test_proportion is None:
            train_val_test_proportion = [0.7, 0.1, 0.2]
        if expand_list:
            tables = {table_name: convert_str_to_list(table) for table_name, table in tables.items()}

        dataset = []
        for table_name, table in tqdm(tables.items()):
            logging.info(f"Generating contrastive samples for table {table_name}")
            if use_selective_negatives or use_selective_positives:
                similarity_df = self.get_similarity_matrix(table, text_embedder, expand_list)
            for col_name, col_values in table.items():
                if expand_list and type(col_values[0]) == list:
                    non_empty_col_values = [str(v) for v_list in col_values for v in v_list]
                else:
                    non_empty_col_values = [str(v) for v in col_values.dropna() if v != "[]"]

                if remove_duplicates:
                    non_empty_col_values = list(set(non_empty_col_values))

                positive_pairs = self._pairwise(non_empty_col_values)
                if use_selective_positives:
                    similarity_scores = similarity_df.loc[non_empty_col_values][non_empty_col_values]
                    similarity_scores = similarity_scores.where(np.tril(np.ones(similarity_scores.shape), -1).astype(bool))
                    similarity_scores = similarity_scores.stack().reset_index()
                    similarity_scores.columns = ["sent0", "sent1", "similarity"]
                    positive_pairs.extend(similarity_scores.sort_values(by="similarity", ascending=True).head(positive_pairs_size)[["sent0", "sent1"]].values)

                for sent0, sent1 in positive_pairs:
                    # get values from columns that are not the current column
                    for other_name, other_values in table.items():
                        if other_name != col_name:
                            if expand_list and type(other_values[0]) == list:
                                non_empty_other_values = [str(v) for v_list in other_values for v in v_list]
                            else:
                                non_empty_other_values = [str(v) for v in other_values.dropna() if v != "[]"]
                            if remove_duplicates:
                                non_empty_other_values = list(set(non_empty_other_values))
                            if len(non_empty_other_values) > hard_neg_size:
                                if use_selective_negatives:
                                    similarity_scores = similarity_df.loc[sent0]
                                    filtered_scores = similarity_scores[similarity_scores.index.isin(non_empty_other_values) & (similarity_scores.index != sent0)]
                                    hard_negs = filtered_scores.sort_values(ascending=False).head(hard_neg_size).index.tolist()
                                else:
                                    hard_negs = random.sample(list(non_empty_other_values), hard_neg_size)
                            else:
                                hard_negs = non_empty_other_values
                            for hard_neg in hard_negs:
                                row = {self.SENT0_COL: sent0, self.SENT1_COL: sent1, self.HARD_NEG_COL: hard_neg}
                                if return_col_table_names:
                                    row[self.POS_COL_NAME] = col_name
                                    row[self.NEG_COL_NAME] = other_name
                                    row[self.TABLE_NAME] = table_name
                                dataset.append(row)

        dataset = pd.DataFrame(dataset).astype(str)
        unique_positive_pairs = dataset[[self.SENT0_COL, self.SENT1_COL]].drop_duplicates() # the same positive pairs cannot appear on train, val and test
        train_df, test_df = train_test_split(unique_positive_pairs, train_size=train_val_test_proportion[0], random_state=self.seed,
                                             shuffle=True)
        val_df, test_df = train_test_split(test_df,
                                           train_size=train_val_test_proportion[1] / sum(train_val_test_proportion[1:]),
                                           random_state=self.seed, shuffle=True)

        train_df = train_df.merge(dataset, on=[self.SENT0_COL, self.SENT1_COL], how="left")
        val_df = val_df.merge(dataset, on=[self.SENT0_COL, self.SENT1_COL], how="left")
        test_df = test_df.merge(dataset, on=[self.SENT0_COL, self.SENT1_COL], how="left")

        train_df = train_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        val_df = val_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        test_df = test_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        return train_df, val_df, test_df

    def _pairwise(self, iterable):
        """
        s -> (s0,s1), (s2,s3), (s4, s5), ...
        :param iterable:
        :return:
        """
        a = iter(iterable)
        return list(zip(a, a))

    def get_similarity_matrix(self, table: pd.DataFrame, text_embedder: HFTextEmbedder, expand_list: bool):
        self.logger.info("Generating similarity matrix")
        unique_values = []
        for col_name, col_values in table.items():
            if expand_list:
                non_empty_col_values = [str(v) for v_list in col_values for v in v_list]
            else:
                non_empty_col_values = [str(v) for v in col_values.dropna() if v != "[]"]
            unique_values.extend(list(set(non_empty_col_values)))
        unique_values = list(set(unique_values))
        embeddings = text_embedder.get_sentence_embeddings(unique_values)
        sim_matrix = cos_sim(embeddings, embeddings)
        similarity_df = pd.DataFrame(sim_matrix.cpu(), index=unique_values, columns=unique_values)
        return similarity_df