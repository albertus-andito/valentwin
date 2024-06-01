import logging
import pandas as pd

from sentence_transformers import SentenceTransformer
from typing import Dict, Tuple, Union

from valentwin.algorithms.deepjoin.column_to_text_transformer import ColumnToTextTransformer
from valentwin.algorithms.match import Match
from valentwin.algorithms.valentwin.utils import cos_sim, euclidean_distances
from valentwin.algorithms import BaseMatcher
from valentwin.data_sources.base_table import BaseTable
from valentwin.embedder.text_embedder import HFTextEmbedder
from valentwin.utils.utils import ColumnTypes


class DeepJoin(BaseMatcher):
    """
    Our implementation of DeepJoin based on the paper:
    Yuyang Dong, Chuan Xiao, Takuma Nozawa, Masafumi Enomoto, and Masafumi Oyamada.
    DeepJoin: Joinable Table Discovery with Pre-trained Language Models. PVLDB, 16(10): 2458 - 2470, 2023.
    doi:10.14778/3603581.3603587
    """

    def __init__(self, model_path: str = None, model: Union[SentenceTransformer, HFTextEmbedder] = None,
                 column_to_text_transformation: str = "title-colname-stat-col",
                 all_tables: Dict[str, pd.DataFrame] = None, similarity_measure="cosine",
                 use_cache: bool = True, device="cuda"):
        if model is None:
            if "simcse" in model_path:
                self.model = HFTextEmbedder(model_name=model_path, device=device)
            else:
                self.model = SentenceTransformer(model_path, device=device)
        else:
            self.model = model
        if all_tables is not None:
            self.column_to_text_transformer = ColumnToTextTransformer(all_tables, self.model.tokenizer)
        self.column_to_text_transformation = column_to_text_transformation
        self.similarity_measure = similarity_measure
        self.device = device if self.model is None else self.model.device

        self.use_cache = use_cache
        self.embedding_cache = {}
        self.logger = logging.getLogger(__name__)

    def get_matches(self, source_input: BaseTable, target_input: BaseTable) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:
        self.logger.info("Generating embeddings...")
        if self.column_to_text_transformer is None:
            column_to_text_transformer = ColumnToTextTransformer({source_input.name: source_input.get_df(),
                                                                      target_input.name: target_input.get_df()},
                                                                     self.model.tokenizer)
        else:
            column_to_text_transformer = self.column_to_text_transformer
        source_embeddings = self._generate_table_embeddings(source_input, column_to_text_transformer)
        target_embeddings = self._generate_table_embeddings(target_input, column_to_text_transformer)

        # Calculate similarities
        self.logger.info("Calculating similarities...")
        if self.similarity_measure == "cosine":
            similarities = cos_sim(source_embeddings, target_embeddings)
        elif self.similarity_measure == "euclidean":
            similarities = euclidean_distances(source_embeddings, target_embeddings, device=self.device)
        else:
            raise ValueError(f"Unknown similarity measure: {self.similarity_measure}")

        matches = {}
        for i, source_column in enumerate(source_input.get_columns()):  # same as in ValenTwin
            for j, target_column in enumerate(target_input.get_columns()):
                if target_column.column_type.value == source_column.column_type.value:
                    column_type = target_column.column_type.value
                else:
                    column_type = ColumnTypes.TEXTUAL.value
                matches.update(Match(target_table_name=target_input.name, target_column_name=target_column.name,
                                     source_table_name=source_input.name, source_column_name=source_column.name,
                                     similarity=similarities[i][j].item(), column_type=column_type).to_dict)

        # Remove the pairs with zero similarity
        matches = {k: v for k, v in matches.items() if v > 0.0}
        # sort matches by similarity
        matches = {k: v for k, v in sorted(matches.items(), key=lambda item: item[1], reverse=True)}

        return matches

    def _generate_table_embeddings(self, table: BaseTable, col_to_text_transformer: ColumnToTextTransformer):
        if self.use_cache:
            if table.name in self.embedding_cache:
                return self.embedding_cache[table.name]

        column_representations = col_to_text_transformer.get_all_column_representations(method=self.column_to_text_transformation,
                                                                                        tables={table.name: table.get_df()})
        flatten_representation = [column_representation for table_name, table_representation in column_representations.items()
                                    for column_name, column_representation in table_representation.items()]
        if isinstance(self.model, HFTextEmbedder):
            embeddings = self.model.get_sentence_embeddings(flatten_representation)
        else:  # SentenceTransformer
            embeddings = self.model.encode(flatten_representation)

        if self.use_cache and table.name not in self.embedding_cache:
            self.embedding_cache[table.name] = embeddings

        return embeddings

