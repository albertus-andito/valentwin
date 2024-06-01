# Ported from Valentine: https://github.com/delftdata/valentine
# Added one_to_one_matches_intra_table

from valentwin.metrics import metrics as metrics_module
from typing import List, Dict, Tuple

metrics = {"names": ["precision", "recall", "f1_score", "precision_at_n_percent", "recall_at_sizeof_ground_truth"],
           "args": {
               "n": [10, 30, 50, 70, 90]
           }}


def all_metrics(matches: Dict[Tuple[Tuple[str, str], Tuple[str, str]], float],
                golden_standard, one_to_one: bool = False, one_to_one_intra_table: bool = True):
    # load and print the specified metrics
    metric_fns = [getattr(metrics_module, met) for met in metrics['names']]

    final_metrics = dict()

    for metric in metric_fns:
        if metric.__name__ in ["precision", "recall", "f1_score"]:
            final_metrics[metric.__name__] = metric(matches, golden_standard, one_to_one, one_to_one_intra_table)
        elif metric.__name__ != "precision_at_n_percent":  # recall_at_sizeof_ground_truth
            final_metrics[metric.__name__] = metric(matches, golden_standard)
        else:
            for n in metrics['args']['n']:
                final_metrics[metric.__name__.replace('_n_', '_' + str(n) + '_')] = metric(matches, golden_standard, n)
    return final_metrics
