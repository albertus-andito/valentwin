from tqdm.auto import tqdm
from transformers.utils import ModelOutput
from typing import List

import numpy as np
import ot
import torch


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output: ModelOutput, attention_mask: torch.Tensor):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def mean_pooling_first_last_avg(model_output, attention_mask):
    if model_output.hidden_states is None:
        raise Exception(
            "The model output should have hidden_states activated by adding the parameter output_hidden_states=True when retrieving the output!")
    first_layer_embeddings = model_output.hidden_states[1]  # [0] is the embedding layer, while [1] should be the first layer
    last_layer_embeddings = model_output.hidden_states[-1]
    avg_first_last_layer_embeddings = torch.mean(torch.stack([first_layer_embeddings, last_layer_embeddings]),
                                                 dim=0)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(avg_first_last_layer_embeddings.size()).float()
    return torch.sum(avg_first_last_layer_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9)


def cls_pooling(model_output):
    return model_output[0][:, 0]


def cls_mlp_pooling(model_output):
    return model_output.pooler_output


# copied from sentence-transformer (SBERT)
def cos_sim(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def euclidean_distances(a: torch.Tensor, b: torch.Tensor, normalize: bool = True, device: str = "cpu"):
    """
    Computes the euclidean distance between a and b.
    :return: Matrix with res[i][j]  = euclidean_distance(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a = a.unsqueeze(1)
    b = b.unsqueeze(0)

    distances = torch.sqrt(torch.sum((a - b) ** 2, 2))

    if normalize:
        return torch.tensor(normalize_matrix(distances.cpu().numpy())).to(device)
    return distances.to(device)


def wasserstein_distance(mu_s, mu_t, xs, xt):
    """
    Compute the Wasserstein distance between two 2D distributions.

    :param mu_s: The weights for the source distribution.
    :param mu_t: The weights for the target distribution.
    :param xs: The source samples (2D array where each row is a sample).
    :param xt: The target samples (2D array where each row is a sample).
    :return: The Wasserstein distance between the two distributions.
    """
    # Compute the cost matrix (Euclidean distance between samples)
    xs_shape = xs.shape
    xt_shape = xt.shape
    if len(xs_shape) == 1:
        xs = xs.reshape(-1, 1)
    if len(xt_shape) == 1:
        xt = xt.reshape(-1, 1)
    cost_matrix = ot.dist(xs, xt, metric='euclidean')

    # Compute the Wasserstein distance
    wasserstein_distance = ot.emd2(mu_s, mu_t, cost_matrix)

    return wasserstein_distance


def earth_movers_distances(a: List[torch.Tensor], b: List[torch.Tensor], device: str = "cpu", normalize: bool = True):
    earth_movers_distances = np.empty((len(a), len(b)))

    for i, embed_a in tqdm(enumerate(a), total=len(a)):
        for j, embed_b in enumerate(b):
            # handle case where length of embeddings is 0
            if len(embed_a) == 0 or len(embed_b) == 0:
                earth_movers_distances[i, j] = float('inf')
                continue
            weights_a = torch.Tensor(np.full(len(embed_a), 1 / len(embed_a)))
            weights_b = torch.Tensor(np.full(len(embed_b), 1 / len(embed_b)))
            distance = wasserstein_distance(weights_a, weights_b, embed_a, embed_b)
            earth_movers_distances[i, j] = distance

    if normalize:
        return torch.tensor(normalize_matrix(earth_movers_distances)).to(device)

    return torch.tensor(earth_movers_distances).to(device)


def normalize_matrix(matrix: np.ndarray):
    min_val = np.min(matrix)
    max_val = np.max(matrix)

    # Perform min-max normalization
    return (matrix - min_val) / (max_val - min_val)


