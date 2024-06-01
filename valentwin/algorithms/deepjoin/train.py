# Here, we use sentence-transformers, not just transformers package, so that it is consistent with the paper.

from sentence_transformers import InputExample, SentenceTransformer, models, losses
from sentence_transformers.util import cos_sim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Union

import logging
import math
import pandas as pd
import random

from valentwin.algorithms.deepjoin.column_to_text_transformer import ColumnToTextTransformer
from valentwin.embedder.text_embedder import HFTextEmbedder

logger = logging.getLogger(__name__)


def construct_train_dataset(all_tables: Dict[str, pd.DataFrame], model_name: str = None,
                            model: Union[SentenceTransformer, HFTextEmbedder] = None,
                            column_to_text_transformation: str = "title-colname-stat-col",
                            shuffle_rate: float = 0.2, seed: int = 42, device="cuda"):
    if model is None:
        model = SentenceTransformer(model_name, device=device)
    col_to_text_transformer = ColumnToTextTransformer(all_tables, model.tokenizer)
    column_representations = col_to_text_transformer.get_all_column_representations(method=column_to_text_transformation)

    # To get the positive pairs, we follow the approach in OmniMatch, where "positive training pairs are generated
    # based on pairwise cosine similarity of initial column representations (more than or equal to 0.9 ot ensure
    # high true positive rates)".
    flatten_representation = [column_representation for table_name, table_representation in column_representations.items()
                                for column_name, column_representation in table_representation.items()]
    if isinstance(model, HFTextEmbedder):
        embeddings = model.get_sentence_embeddings(flatten_representation)
    else:  # SentenceTransformer
        embeddings = model.encode(flatten_representation)
    similarities = cos_sim(embeddings, embeddings)

    positive_pairs = set()
    positive_pairs_indices = set()
    for i in range(len(similarities) - 1):
        for j in range(i + 1, len(similarities)):
            sim = similarities[i][j].item()
            if sim >= 0.9 and (i, j) not in positive_pairs_indices and (j, i) not in positive_pairs_indices:
                positive_pairs_indices.add((i, j))
                positive_pairs.add((flatten_representation[i], flatten_representation[j]))  # (X, Y)

    if shuffle_rate > 0:
        # sample from the positive pairs
        random.seed(seed)
        to_be_shuffled = random.sample(positive_pairs_indices, int(shuffle_rate * len(positive_pairs_indices)))

        shuffled_column_representations = col_to_text_transformer.get_all_column_representations(method=column_to_text_transformation,
                                                                                                 shuffle_column_values=True)
        shuffled_flatten_representation = [column_representation for table_name, table_representation in shuffled_column_representations.items()
                                for column_name, column_representation in table_representation.items()]

        for i, j in to_be_shuffled:
            positive_pairs.add((shuffled_flatten_representation[i], flatten_representation[j])) # (X', Y)

    return list(positive_pairs)


def train_model(model_name: str, train_dataset: List[Tuple[str, str]], model_save_path: str = None, batch_size: int = 32,
                learning_rate: float = 2e-5, warmup_steps: int = None, weight_decay: float = 0.01, num_epochs: int = 3,
                device="cuda"):
    # Here we define our SentenceTransformer model
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode="mean")
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

    # Special data loader that avoid duplicates within a batch
    train_dataset = [InputExample(texts=[a, b], label=1) for a, b in train_dataset]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Configure the training
    if warmup_steps is None:
        warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
    logging.info(f"Warmup-steps: {warmup_steps}, "
                 f"Weight-decay: {weight_decay}, "
                 f"Learning-rate: {learning_rate}, "
                 f"Batch-size: {batch_size}, "
                 f"Num-epochs: {num_epochs}")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        optimizer_params={"lr": learning_rate},
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        output_path=model_save_path,
        use_amp=True,  # Set to True, if your GPU supports FP16 operations
    )
    logging.info("Training completed.")
    return model

