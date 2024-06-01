import logging
import pandas as pd
import torch

from torch import Tensor
from typing import Dict, List, Sequence, Tuple, Union

from valentwin.algorithms import BaseMatcher
from valentwin.algorithms.match import Match
from valentwin.embedder.text_embedder import TextEmbedder
from valentwin.algorithms.valentwin.utils import cos_sim, euclidean_distances, earth_movers_distances
from valentwin.data_sources.base_table import BaseTable
from valentwin.data_sources import DataframeTable
from valentwin.utils.utils import ColumnTypes


class ValenTwin(BaseMatcher):

    def __init__(self, text_embedder: TextEmbedder, column_name_weight: float = 0.3,
                 measure: str = "cosine_similarity",
                 column_name_measure: str = None):
        self.text_embedder = text_embedder
        self.measure = measure

        self.column_name_weight = column_name_weight
        self.column_data_weight = 1 - column_name_weight
        self.column_name_measure = column_name_measure if column_name_measure is not None else measure

        self.logger = logging.getLogger(__name__)

    def get_matches_from_batch(self, tables: Sequence[pd.DataFrame], table_names: List[str]) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:
        if table_names is None:
            table_names = [f"table_{i}" for i in range(len(tables))]
        tables = [DataframeTable(table, name=name) for table, name in zip(tables, table_names)]

        self.logger.info("Generating embeddings...")
        embeddings = self._generate_batch_embeddings(tables)
        column_name_embeddings = torch.cat([self.text_embedder.get_sentence_embeddings([col.name])
                                            for table in tables for col in table.get_columns()], dim=0)

        self.logger.info("Calculating similarities/distances...")
        matrix = self._calculate_matrix(embeddings, embeddings, column_name_embeddings, column_name_embeddings)

        matches = {}
        if isinstance(matrix, Tensor):
            matrix = matrix.cpu()
        table_columns = [(table.name, col.name, col.column_type.value) for table in tables for col in table.get_columns()]
        for i, (source_table_name, source_column_name, source_column_type) in enumerate(table_columns):
            for j, (target_table_name, target_column_name, target_column_type) in enumerate(table_columns):
                if source_table_name == target_table_name:
                    continue
                if target_column_type == source_column_type:
                    column_type = target_column_type
                else:
                    column_type = ColumnTypes.TEXTUAL.value
                matches.update(Match(target_table_name=target_table_name, target_column_name=target_column_name,
                                     source_table_name=source_table_name, source_column_name=source_column_name,
                                     similarity=matrix[i][j].item(), column_type=column_type).to_dict)

        # Remove the pairs with zero similarity
        matches = {k: v for k, v in matches.items() if v > 0.0}
        # sort matches by similarity
        matches = {k: v for k, v in sorted(matches.items(), key=lambda item: item[1], reverse=True)}

        return matches

    def _generate_batch_embeddings(self, tables: List[BaseTable]) -> Union[torch.Tensor, List[torch.Tensor]]:
        embeddings = []
        for table in tables:
            if self.measure == "cosine_similarity" or self.measure == "euclidean_distance":
                table_embeddings = torch.stack([self.text_embedder.get_aggregated_embeddings(col.data)
                                                for col in table.get_columns()], dim=0)
            else:
                table_embeddings = [self.text_embedder.get_sentence_embeddings(col.data) for col in table.get_columns()]
            embeddings.append(table_embeddings)
        if self.measure == "cosine_similarity" or self.measure == "euclidean_distance":
            embeddings = torch.cat(embeddings, dim=0)
        else:
            embeddings = [column_embeddings for table_embeddings in embeddings for column_embeddings in table_embeddings]
        return embeddings

    def get_matches(self, source_input: BaseTable, target_input: BaseTable) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:
        self.logger.info("Generating embeddings...")
        source_embeddings, target_embeddings, source_column_name_embeddings, target_column_name_embeddings = self._generate_embeddings(source_input, target_input)

        self.logger.info("Calculating similarities/distances...")
        matrix = self._calculate_matrix(source_embeddings, target_embeddings, source_column_name_embeddings, target_column_name_embeddings)

        matches = {}
        if isinstance(matrix, Tensor):
            matrix = matrix.cpu()
        for i, source_column in enumerate(source_input.get_columns()):
            for j, target_column in enumerate(target_input.get_columns()):
                if target_column.column_type.value == source_column.column_type.value:
                    column_type = target_column.column_type.value
                else:
                    column_type = ColumnTypes.TEXTUAL.value
                matches.update(Match(target_table_name=target_input.name, target_column_name=target_column.name,
                                     source_table_name=source_input.name, source_column_name=source_column.name,
                                     similarity=matrix[i][j].item(), column_type=column_type).to_dict)

        # Remove the pairs with zero similarity
        matches = {k: v for k, v in matches.items() if v > 0.0}
        # sort matches by similarity
        matches = {k: v for k, v in sorted(matches.items(), key=lambda item: item[1], reverse=True)}

        return matches

    def _generate_embeddings(self, source_input: BaseTable, target_input: BaseTable) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.measure == "cosine_similarity" or self.measure == "euclidean_distance":
            source_embeddings = torch.stack(
                [self.text_embedder.get_aggregated_embeddings(col.data) for col in source_input.get_columns()], dim=0)
            target_embeddings = torch.stack(
                [self.text_embedder.get_aggregated_embeddings(col.data) for col in target_input.get_columns()], dim=0)
        else:
            source_embeddings = [self.text_embedder.get_sentence_embeddings(col.data) for col in
                                 source_input.get_columns()]
            target_embeddings = [self.text_embedder.get_sentence_embeddings(col.data) for col in
                                 target_input.get_columns()]

        if self.column_name_weight > 0.0:
            source_column_name_embeddings, target_column_name_embeddings = self._generate_column_name_embeddings(source_input, target_input)
        else:
            source_column_name_embeddings = None
            target_column_name_embeddings = None

        return source_embeddings, target_embeddings, source_column_name_embeddings, target_column_name_embeddings

    def _generate_column_name_embeddings(self, source_input: BaseTable, target_input: BaseTable) -> Tuple[torch.Tensor, torch.Tensor]:
        source_column_name_embeddings = [self.text_embedder.get_sentence_embeddings([col.name]) for col in
                                         source_input.get_columns()]
        target_column_name_embeddings = [self.text_embedder.get_sentence_embeddings([col.name]) for col in
                                         target_input.get_columns()]
        if self.column_name_measure == "cosine_similarity" or self.column_name_measure == "euclidean_distance":
            source_column_name_embeddings = torch.cat(source_column_name_embeddings, dim=0)
            target_column_name_embeddings = torch.cat(target_column_name_embeddings, dim=0)
        return source_column_name_embeddings, target_column_name_embeddings

    def _calculate_matrix(self, source_embeddings: Union[torch.Tensor, List[torch.Tensor], List[List[float]]],
                          target_embeddings: Union[torch.Tensor, List[torch.Tensor], List[List[float]]],
                          source_column_name_embeddings: torch.Tensor, target_column_name_embeddings: torch.Tensor,
                          measure: str = None) -> torch.Tensor:
        content_matrix = self.calculate_similarity_or_distance(source_embeddings, target_embeddings, measure=measure).to(self.text_embedder.device)

        matrix = content_matrix * self.column_data_weight

        if source_column_name_embeddings is not None and target_column_name_embeddings is not None:
            column_name_matrix = self.calculate_similarity_or_distance(source_column_name_embeddings,
                                                                       target_column_name_embeddings,
                                                                       measure=self.column_name_measure).to(self.text_embedder.device)

            # Identify zero vectors in context embeddings
            # (in the unlikely event of a column has no name, it is a zero vector)
            if self.column_name_measure == "cosine_similarity" or self.column_name_measure == "euclidean_distance":
                zero_vecs_1 = torch.all(source_column_name_embeddings == 0, dim=1)
                zero_vecs_2 = torch.all(target_column_name_embeddings == 0, dim=1)
            else:
                zero_vecs_1 = torch.tensor([torch.all(tensor == 0).item() for tensor in source_column_name_embeddings])
                zero_vecs_2 = torch.tensor([torch.all(tensor == 0).item() for tensor in target_column_name_embeddings])
            zero_vec_pairs = torch.outer(zero_vecs_1, zero_vecs_2)
            # For those zero context pairs, set context similarity to 0
            column_name_matrix[zero_vec_pairs] = 0
            # Weighted addition
            matrix += column_name_matrix * torch.tensor(self.column_name_weight).to(self.text_embedder.device)
            # For zero context pairs, assign content similarity as final similarity
            # sims[zero_vec_pairs] += content_sims[zero_vec_pairs] * context_weight
            matrix[zero_vec_pairs] += column_name_matrix[zero_vec_pairs]

        return matrix

    def calculate_similarity_or_distance(self, embeddings_1, embeddings_2, measure: str = None):
        if measure is None:
            measure = self.measure
        if measure == "cosine_similarity":
            matrix = cos_sim(embeddings_1, embeddings_2)
        elif measure == "euclidean_distance":
            matrix = euclidean_distances(embeddings_1, embeddings_2)
        elif measure == "earth_movers_distance":
            matrix = earth_movers_distances(embeddings_1, embeddings_2, device=self.text_embedder.device, normalize=True)
        else:
            raise ValueError(f"Unknown alignment metric {measure}")

        if measure in ["euclidean_distance", "earth_movers_distance"]:
            matrix = 1 - matrix
        return matrix