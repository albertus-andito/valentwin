# Ported from Starmie: https://github.com/megagonlabs/starmie

import torch
import numpy as np
import sklearn.metrics as metrics

from collections import deque


def evaluate(model, iterator, threshold=None):
    """Evaluate a model on a validation/test dataset

    Args:
        model (TableModel): the EM model
        iterator (Iterator): the valid/test dataset iterator
        threshold (float, optional): the threshold on the 0-class

    Returns:
        float: the F1 score
        float (optional): if threshold is not provided, the threshold
            value that gives the optimal F1
    """
    all_p = []
    all_y = []
    all_probs = []
    with torch.no_grad():
        for batch in iterator:
            if len(batch) == 4:
                x1, x2, x12, y = batch
                logits = model(x1, x2, x12)
            else:
                x, y = batch
                logits = model(x)

            # print(probs)
            probs = logits.softmax(dim=1)[:, 1]

            # print(logits)
            # pred = logits.argmax(dim=1)
            all_probs += probs.cpu().numpy().tolist()
            # all_p += pred.cpu().numpy().tolist()
            all_y += y.cpu().numpy().tolist()

    if threshold is not None:
        pred = [1 if p > threshold else 0 for p in all_probs]
        f1 = metrics.f1_score(all_y, pred)
        return f1
    else:
        best_th = 0.5
        f1 = 0.0  # metrics.f1_score(all_y, all_p)

        for th in np.arange(0.0, 1.0, 0.05):
            pred = [1 if p > th else 0 for p in all_probs]
            new_f1 = metrics.f1_score(all_y, pred)
            if new_f1 > f1:
                f1 = new_f1
                best_th = th

        return f1, best_th


def connected_components(pairs, cluster_size=50):
    """Helper function for computing the connected components
    """
    edges = {}
    for left, right, _ in pairs:
        if left not in edges:
            edges[left] = []
        if right not in edges:
            edges[right] = []

        edges[left].append(right)
        edges[right].append(left)

    print('num nodes =', len(edges))
    all_ccs = []
    used = set([])
    for start in edges:
        if start in used:
            continue
        used.add(start)
        cc = [start]

        queue = deque([start])
        while len(queue) > 0:
            u = queue.popleft()
            for v in edges[u]:
                if v not in used:
                    cc.append(v)
                    used.add(v)
                    queue.append(v)

            if len(cc) >= cluster_size:
                break

        all_ccs.append(cc)
        # print(cc)
    return all_ccs

