import argparse
import logging
import numpy as np
import random
import torch

from valentwin.algorithms.starmie.dataset import PretrainTableDataset
from valentwin.algorithms.starmie.pretrain import train

logging.basicConfig(level=logging.INFO)
logging.getLogger(__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="Path to tables to train on")
    parser.add_argument("--model_path", type=str, default="results/", help="Path to save the model")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--lm", type=str, default='roberta-base')
    parser.add_argument("--projector", type=int, default=768)
    parser.add_argument("--augment_ops", nargs="+", type=str, default=['drop_col,sample_row'])
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    # single-column mode without table context
    parser.add_argument("--single_column", dest="single_column", action="store_true")
    # row / column-ordered for preprocessing
    parser.add_argument("--table_orders", nargs="+", type=str, default=['column'])
    # for sampling
    parser.add_argument("--sample_meths", nargs="+", type=str, default=['head'])

    hp = parser.parse_args()

    if "all" in hp.augment_ops:
        augment_ops = ["drop_col", "sample_row", "sample_row_ordered", "shuffle_col", "drop_cell",
                       "sample_cells", "replace_cells", "drop_head_cells", "drop_num_cells", "swap_cells",
                       "drop_num_col", "drop_nan_col"]
    else:
        augment_ops = hp.augment_ops
    if "all" in hp.sample_meths:
        sample_meths = ["head", "alphaHead", "random", "constant", "frequent", "tfidf_token", "tfidf_entity",
                        "tfidf_row"]
    else:
        sample_meths = hp.sample_meths
    if "all" in hp.table_orders:
        table_orders = ["column", "row"]
    else:
        table_orders = hp.table_orders

    for augment_op in augment_ops:
        for sample_meth in sample_meths:
            for table_order in table_orders:
                logger.info("Training model with augment_op: %s, sample_meth: %s, table_order: %s" % (augment_op, sample_meth, table_order))
                hp.table_order = table_order
                hp.augment_op = augment_op
                hp.sample_meth = sample_meth

                # set seeds
                seed = hp.run_id
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                path = hp.task
                trainset = PretrainTableDataset.from_hp(path, hp)

                model = train(trainset, hp)

    logger.info("Training completed.")
