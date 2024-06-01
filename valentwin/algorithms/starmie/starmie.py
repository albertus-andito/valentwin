import logging
import torch

from argparse import Namespace
from typing import Dict, Tuple

from valentwin.algorithms.match import Match
from valentwin.algorithms.valentwin.utils import cos_sim, euclidean_distances
from valentwin.algorithms.starmie.dataset import PretrainTableDataset
from valentwin.algorithms.starmie.model import BarlowTwinsSimCLR
from valentwin.algorithms.starmie.pretrain import load_checkpoint
from valentwin.algorithms import BaseMatcher
from valentwin.data_sources.base_table import BaseTable
from valentwin.utils.utils import ColumnTypes


class Starmie(BaseMatcher):

    def __init__(self, model_path: str = None, model: BarlowTwinsSimCLR = None, hp: Namespace = None,
                 similarity_measure="cosine", use_cache: bool = True):
        self.model_path = model_path
        if model_path is not None:
            self.model, hp = load_checkpoint(model_path)
        else:
            self.model = model
            hp = hp
        # There's really no reason to create this object below, other than to get the tokenize and pad methods
        self.pretrain_table_dataset = PretrainTableDataset.from_hp(hp.task, hp)
        self.similarity_measure = similarity_measure

        self.use_cache = use_cache
        self.embedding_cache = {}
        self.logger = logging.getLogger(__name__)

    def get_matches(self, source_input: BaseTable, target_input: BaseTable) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:
        self.logger.info("Generating embeddings...")
        source_embeddings = self._generate_table_embeddings(source_input)
        target_embeddings = self._generate_table_embeddings(target_input)

        # Calculate similarities
        self.logger.info("Calculating similarities...")
        if self.similarity_measure == "cosine":
            similarities = cos_sim(source_embeddings, target_embeddings)
        elif self.similarity_measure == "euclidean":
            similarities = euclidean_distances(source_embeddings, target_embeddings)
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

    def _generate_table_embeddings(self, table: BaseTable):
        if self.use_cache:
            if table.name in self.embedding_cache:
                return self.embedding_cache[table.name]

        x, _ = self.pretrain_table_dataset._tokenize(table.get_df())
        batch = [(x, x, [])]
        with torch.no_grad():
            x, _, _ = self.pretrain_table_dataset.pad(batch)
            column_vectors = self.model.inference(x)

        if self.use_cache and table.name not in self.embedding_cache:
            self.embedding_cache[table.name] = column_vectors

        return column_vectors

