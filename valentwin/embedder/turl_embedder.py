import numpy as np
import pandas as pd
import os
import torch

from valentwin.embedder.turl.data_loader.hybrid_data_loaders import WikiHybridTableDataset
from valentwin.embedder.turl.model.configuration import TableConfig
from valentwin.embedder.turl.model.model import HybridTableMaskedLM
from valentwin.embedder.turl.model.transformers import BertTokenizer
from valentwin.embedder.turl.utils import load_entity_vocab


def get_entity_id(entity: str, entity_vocab: dict):
    if entity in entity_vocab:
        return entity_vocab[entity]['wiki_id']
    else:
        return -1


def process_dataset_get_entity_ids(table_fpath: str, output_fpath: str, data_dir: str = "data/wikitables_v2/"):
    entity_vocab = load_entity_vocab(data_dir, ignore_bad_title=True, min_ent_count=2, title_as_key=True)
    table_df = pd.read_csv(table_fpath)
    for col in table_df.columns:
        table_df[col] = table_df[col].apply(get_entity_id, args=(entity_vocab,))
    table_df.to_csv(output_fpath)


class TurlEmbedder:
    """
    Our attempt to use TURL embeddings for ALITE. This is pretty much our estimation of how the TURL embeddings are used
    since it is not explained in detail in the paper.
    The performance using this method is quite poor, so we are not using it in the final implementation.
    """

    MODEL_CLASSES = {
        'CF': (TableConfig, HybridTableMaskedLM, BertTokenizer),
    }

    def __init__(self, data_dir: str = "data/wikitables_v2/",
                 config_name: str = "configs/table-base-config_v2.json",
                 model_name_or_path: str = "output/hybrid/v2/model_v1_table_0.2_0.6_0.7_10000_1e-4_candnew_0_adam/",
                 device: str = 'cpu'):
        super().__init__()
        self.data_dir = data_dir
        self.config_name = config_name
        self.device = torch.device(device)

        # load entity vocab from entity_vocab.txt
        self.entity_vocab = load_entity_vocab(data_dir, ignore_bad_title=True, min_ent_count=2)
        self.entity_wikid2id = {self.entity_vocab[x]['wiki_id']: x for x in self.entity_vocab}
        self.entity_wikititle2wikid = {self.entity_vocab[x]['wiki_title']: self.entity_vocab[x]['wiki_id'] for x in self.entity_vocab}

        config_class, model_class, _ = self.MODEL_CLASSES['CF']
        config = config_class.from_pretrained(config_name)
        config.output_attentions = True

        # For CF, we use the base HybridTableMaskedLM, and directly load the pretrained checkpoint
        checkpoint = model_name_or_path
        self.model = model_class(config, is_simple=True)
        checkpoint = torch.load(os.path.join(checkpoint, 'pytorch_model.bin'), map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        # load the module for cell filling baselines
        # CF = cell_filling(data_dir)

    def get_table_column_embeddings(self, table_df: pd.DataFrame, use_header: bool = False):
        headers = table_df.columns.tolist()
        dataset = WikiHybridTableDataset(self.data_dir, self.entity_vocab, max_cell=100, max_input_tok=350,
                                         max_input_ent=150, src="dev", max_length=[50, 10, 10], force_new=False,
                                         tokenizer=None, mode=0)
        sample_size = 50 if len(table_df) > 50 else len(table_df)
        table_embeddings = {}
        for header in headers:
            prev_ent_embeddings = None
            for i in range(0, len(table_df), sample_size):
                core_entities_text = table_df[header][i:i+sample_size].astype(str).tolist()
                core_entities = [self.entity_wikititle2wikid.get(x, -1) for x in core_entities_text]
                input_tok, input_tok_type, input_tok_pos, input_tok_mask, \
                    input_ent, input_ent_text, input_ent_text_length, input_ent_type, input_ent_mask_type, input_ent_mask, \
                    candidate_entity_set = self.build_input(-1, "", "", None,
                                                            [header] if use_header else [],
                                                            core_entities, core_entities_text, None, dataset)

                input_tok = input_tok.to(self.device)
                input_tok_type = input_tok_type.to(self.device)
                input_tok_pos = input_tok_pos.to(self.device)
                # input_tok_mask = input_tok_mask.to(self.device)
                input_ent_text = input_ent_text.to(self.device)
                input_ent_text_length = input_ent_text_length.to(self.device)
                input_ent = input_ent.to(self.device)
                input_ent_type = input_ent_type.to(self.device)
                input_ent_mask_type = input_ent_mask_type.to(self.device)
                # input_ent_mask = input_ent_mask.to(self.device)
                candidate_entity_set = candidate_entity_set.to(self.device)

                with torch.no_grad():
                    tok_outputs, ent_outputs = self.model(input_tok, input_tok_type, input_tok_pos, input_tok_mask,
                                                     input_ent_text, input_ent_text_length, input_ent_mask_type,
                                                     input_ent, input_ent_type, input_ent_mask, candidate_entity_set)

                ent_outputs_only = ent_outputs[1][0, 1:sample_size+1]  # get only the embeddings of entities
                current_ent_embeddings = torch.mean(ent_outputs_only, dim=0)
                if prev_ent_embeddings is not None:
                    if torch.dist(prev_ent_embeddings, current_ent_embeddings) < 0.05:  #convergence
                        table_embeddings[header] = prev_ent_embeddings
                        break
                    else:  #combine with previous embeddings, but what is "combine"? sum or mean?
                        current_ent_embeddings = (prev_ent_embeddings + current_ent_embeddings) / 2
                        # current_ent_embeddings = (prev_ent_embeddings + current_ent_embeddings)
                prev_ent_embeddings = current_ent_embeddings
            table_embeddings[header] = prev_ent_embeddings
        return table_embeddings


    def build_input(self, pgEnt, pgTitle, secTitle, caption, headers, core_entities, core_entities_text, entity_cand,
                       config):
        tokenized_pgTitle = config.tokenizer.encode(pgTitle, max_length=config.max_title_length,
                                                    add_special_tokens=False)
        tokenized_meta = tokenized_pgTitle + \
                         config.tokenizer.encode(secTitle, max_length=config.max_title_length, add_special_tokens=False)
        if caption is not None and caption != secTitle:
            tokenized_meta += config.tokenizer.encode(caption, max_length=config.max_title_length,
                                                      add_special_tokens=False)
        tokenized_headers = [
            config.tokenizer.encode(header, max_length=config.max_header_length, add_special_tokens=False)
            for header in headers]
        input_tok = []
        input_tok_pos = []
        input_tok_type = []
        tokenized_meta_length = len(tokenized_meta)
        input_tok += tokenized_meta
        input_tok_pos += list(range(tokenized_meta_length))
        input_tok_type += [0] * tokenized_meta_length
        header_span = []
        for tokenized_header in tokenized_headers:
            tokenized_header_length = len(tokenized_header)
            header_span.append([len(input_tok), len(input_tok) + tokenized_header_length])
            input_tok += tokenized_header
            input_tok_pos += list(range(tokenized_header_length))
            input_tok_type += [1] * tokenized_header_length

        input_ent = []
        input_ent_text = []
        input_ent_type = []
        if pgEnt is not None:
            input_ent += [config.entity_wikid2id[pgEnt] if pgEnt != -1 else 0]
            input_ent_text += [tokenized_pgTitle[:config.max_cell_length]]
            input_ent_type += [2]

        # core entities in the subject column
        input_ent += [config.entity_wikid2id[entity] if entity != -1 else 0 for entity in core_entities] # TODO, check whether it should be -1 or 0
        input_ent_text += [
            config.tokenizer.encode(entity_text, max_length=config.max_cell_length, add_special_tokens=False) if len(
                entity_text) != 0 else [] for entity_text in core_entities_text]
        input_ent_type += [3] * len(core_entities)

        # append [ent_mask]
        input_ent += [config.entity_wikid2id['[ENT_MASK]']] * len(core_entities)
        input_ent_text += [[]] * len(core_entities)
        input_ent_type += [4] * len(core_entities)

        input_ent_cell_length = [len(x) if len(x) != 0 else 1 for x in input_ent_text]
        max_cell_length = max(input_ent_cell_length)
        input_ent_text_padded = np.zeros([len(input_ent_text), max_cell_length], dtype=int)
        for i, x in enumerate(input_ent_text):
            input_ent_text_padded[i, :len(x)] = x
        assert len(input_ent) == 1 + 2 * len(core_entities)

        # input_tok_mask = np.ones([1, len(input_tok), len(input_tok) + len(input_ent)], dtype=int)
        # input_tok_mask[0, header_span[0][0]:header_span[0][1], len(input_tok) + 1 + len(core_entities):] = 0
        # input_tok_mask[0, header_span[1][0]:header_span[1][1],
        # len(input_tok) + 1:len(input_tok) + 1 + len(core_entities)] = 0
        # input_tok_mask[0, :, len(input_tok) + 1 + len(core_entities):] = 0

        # build the mask for entities
        # input_ent_mask = np.ones([1, len(input_ent), len(input_tok) + len(input_ent)], dtype=int)
        # input_ent_mask[0, 1:1 + len(core_entities), header_span[1][0]:header_span[1][1]] = 0
        # input_ent_mask[0, 1:1 + len(core_entities), len(input_tok) + 1 + len(core_entities):] = np.eye(
        #     len(core_entities),
        #     dtype=int)
        # input_ent_mask[0, 1 + len(core_entities):, header_span[0][0]:header_span[0][1]] = 0
        # input_ent_mask[0, 1 + len(core_entities):, len(input_tok) + 1:len(input_tok) + 1 + len(core_entities)] = np.eye(
        #     len(core_entities), dtype=int)
        # input_ent_mask[0, 1 + len(core_entities):, len(input_tok) + 1 + len(core_entities):] = np.eye(
        #     len(core_entities),
        #     dtype=int)

        # input_tok_mask = torch.LongTensor(input_tok_mask)
        # input_ent_mask = torch.LongTensor(input_ent_mask)
        input_tok_mask = None
        input_ent_mask = None

        input_tok = torch.LongTensor([input_tok])
        input_tok_type = torch.LongTensor([input_tok_type])
        input_tok_pos = torch.LongTensor([input_tok_pos])

        input_ent = torch.LongTensor([input_ent])
        input_ent_text = torch.LongTensor([input_ent_text_padded])
        input_ent_cell_length = torch.LongTensor([input_ent_cell_length])
        input_ent_type = torch.LongTensor([input_ent_type])

        input_ent_mask_type = torch.zeros_like(input_ent)
        input_ent_mask_type[:, 1 + len(core_entities):] = config.entity_wikid2id['[ENT_MASK]']

        # candidate_entity_set = [config.entity_wikid2id[entity] for entity in entity_cand]
        candidate_entity_set = []
        candidate_entity_set = torch.LongTensor([candidate_entity_set])

        return input_tok, input_tok_type, input_tok_pos, input_tok_mask, \
            input_ent, input_ent_text, input_ent_cell_length, input_ent_type, input_ent_mask_type, input_ent_mask, candidate_entity_set

