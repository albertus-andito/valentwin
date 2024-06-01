# Ported implementation from https://github.com/northeastern-datalab/alite
# Aamod Khatiwada, Roee Shraga, Wolfgang Gatterbauer, and Renée J. Miller.
# Integrating Data Lake Tables. PVLDB, 16(4): 932 - 945, 2022

# Here we make some modifications to make sure that columns that are from the same table are not matched
# We also added our implementation of using our TextEmbedder, instead of pre-computed embeddings

import itertools
import json
import numpy as np
import pandas as pd
import torch

from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from typing import Dict, Sequence, Tuple

from valentwin.algorithms import BaseMatcher
from valentwin.algorithms.match import Match
from valentwin.data_sources import DataframeTable
from valentwin.data_sources.base_table import BaseTable
from valentwin.embedder.text_embedder import TextEmbedder
from valentwin.embedder.turl_embedder import TurlEmbedder
from valentwin.utils.utils import get_column_type, ColumnTypes


def find_subsets(s, n):
    return list(itertools.combinations(s, n))


class ALITE(BaseMatcher):
    def __init__(self, text_embedder: TextEmbedder = None, table_embedder: TurlEmbedder = None,
                 pre_computed_embeddings_file_path: str = None, column_names_map_file_path: str = None,
                 clustering_distance_metric: str = "l2"):
        if text_embedder is None and table_embedder is None and pre_computed_embeddings_file_path is None:
            raise ValueError("Either text_embedder, table_embedder, or path to pre-computed embeddings directory must be provided")
        self.text_embedder = text_embedder
        self.table_embedder = table_embedder

        if pre_computed_embeddings_file_path is not None:
            column_names_map = pd.read_csv(column_names_map_file_path)
            self.pre_computed_embeddings = {}
            with open(pre_computed_embeddings_file_path) as f:
                data = json.load(f)
                for table in data:
                    self.pre_computed_embeddings[table.replace(".csv", "")] = {}
                    for col_name, col_data in data[table].items():
                        cols = column_names_map[(column_names_map["table_name"] == table) & (column_names_map["groundtruth_header"].str.lstrip().str.strip() == col_name.lstrip().strip().replace(" (or equivalent)", ""))]
                        if len(cols) > 0:
                            col_name = cols.iloc[0]["column_header"]
                            if type(col_name) is str:
                                col_name = col_name.lstrip().strip().replace("â\x80\x99", "’")
                        else:
                            col_name = col_name.lstrip().strip().replace("â\x80\x99", "’")
                        if "turl" in pre_computed_embeddings_file_path:
                            self.pre_computed_embeddings[table.replace(".csv", "")][str(col_name)] = col_data["entities_only"]
                        else:
                            self.pre_computed_embeddings[table.replace(".csv", "")][str(col_name)] = col_data
            if "bert" in pre_computed_embeddings_file_path:
                self.vec_length = 768
            elif "fasttext" in pre_computed_embeddings_file_path:
                self.vec_length = 300
            else:
                self.vec_length = 312

        self.clustering_distance_metric = clustering_distance_metric

    def get_matches(self, source_input: BaseTable, target_input: BaseTable) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:
        raise NotImplementedError

    def get_matches_from_batch(self, tables: Sequence[pd.DataFrame], table_names: Sequence[str] = None) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:
        if table_names is None:
            table_names = [f"table_{i}" for i in range(len(tables))]
        tables = [DataframeTable(table, name=name) for table, name in zip(tables, table_names)]

        table_embeddings = []
        if self.table_embedder is not None:
            table_embeddings = [self.table_embedder.get_table_column_embeddings(table.get_df()) for table in tables]

        column_embeddings = []
        track_columns = {}  # for debugging only
        track_tables = {}
        i = 0
        all_columns = set()
        total_columns = 0
        for t, table in enumerate(tables):
            for column in table.get_columns():
                print(table.name, column.name)
                column_type = get_column_type(column.data)
                if column_type == ColumnTypes.TEXTUAL: # if it's a textual column
                    all_columns.add(column)
                    total_columns += 1
                    if self.text_embedder is not None:
                        embeddings = self.text_embedder.get_aggregated_embeddings(column.data)
                    elif self.table_embedder is not None:  # use TURL table embeddings
                        embeddings = table_embeddings[t][column.name]
                    else: # use pre-computed embeddings
                        if len(self.pre_computed_embeddings[table.name]) == 0:
                            embeddings = torch.tensor(np.random.uniform(-1, 1, self.vec_length)) # this is stupid, why would you use random vectors???
                        else:
                            embeddings = torch.tensor(self.pre_computed_embeddings[table.name][str(column.name)])
                    column_embeddings.append(embeddings)
                    track_columns[i] = (table.name, column.name)
                    if table.name in track_tables:
                        track_tables[table.name].add(i)
                    else:
                        track_tables[table.name] = {i}

                    i += 1

        column_embeddings = [embedding.cpu() for embedding in column_embeddings]
        x = np.array(column_embeddings)
        zero_positions = set()
        for table in track_tables:
            indices = track_tables[table]
            all_combinations = find_subsets(indices, 2)
            for each in all_combinations:
                zero_positions.add(each)

        arr = np.zeros((len(track_columns), len(track_columns)))
        for i in range(0, len(track_columns) - 1):
            for j in range(i + 1, len(track_columns)):
                # print(i, j)
                if (i, j) not in zero_positions and (j, i) not in zero_positions and i != j:
                    arr[i][j] = 1
                    arr[j][i] = 1
        # convert to sparse matrix representation
        s = csr_matrix(arr)

        all_distance = {}
        all_labels = {}
        min_k = 1
        max_k = 0
        record_result_edges = {}

        for item in track_tables:
            # print(item, len(track_tables[item]))
            if len(track_tables[item]) > min_k:
                min_k = len(track_tables[item])
            max_k += len(track_tables[item])

        for i in range(min_k, min(max_k, max_k)):
            # clusters = KMeans(n_clusters=14).fit(x)
            clusters = AgglomerativeClustering(n_clusters=i, metric=self.clustering_distance_metric,
                                               compute_distances=True, linkage='complete', connectivity=s)
            clusters.fit_predict(x)
            labels = (clusters.labels_)  # .tolist()
            all_labels[i] = labels.tolist()
            all_distance[i] = metrics.silhouette_score(x, labels)
            result_dict = {}
            wrong_results = set()
            for (col_index, label) in enumerate(all_labels[i]):
                if label in result_dict:
                    result_dict[label].add(col_index)
                else:
                    result_dict[label] = {col_index}

            all_result_edges = set()
            for col_index_set in result_dict:
                set1 = result_dict[col_index_set]
                set2 = result_dict[col_index_set]
                current_result_edges = set()
                for s1 in set1:
                    for s2 in set2:
                        current_result_edges.add(tuple(sorted((s1, s2))))
                all_result_edges = all_result_edges.union(current_result_edges)

            record_result_edges[i] = all_result_edges
        distance_list = all_distance.items()
        distance_list = sorted(distance_list)
        algorithm_k = max(all_distance, key=all_distance.get)
        print("The optimal number of clusters is: ", algorithm_k)
        print("The silhouette score is: ", all_distance[algorithm_k])

        # new code for ValenTwin
        matches = {}
        for edge_0, edge_1 in record_result_edges[algorithm_k]:
            if edge_0 != edge_1: # if not the same column
                target_table_name = track_columns[edge_0][0]
                target_column_name = track_columns[edge_0][1]
                source_table_name = track_columns[edge_1][0]
                source_column_name = track_columns[edge_1][1]
                if target_table_name != source_table_name: # if not from the same table
                    matches.update(Match(target_table_name=target_table_name,
                                         target_column_name=target_column_name,
                                         source_table_name=source_table_name,
                                         source_column_name=source_column_name,
                                         similarity=1.0).to_dict)
        return matches
