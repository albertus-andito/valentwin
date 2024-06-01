from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.utils import ModelOutput
from typing import List

import logging
import torch
import torch.nn.functional as F


class TextEmbedder(ABC):

    def __init__(self):
        self.device = "cpu"

    @abstractmethod
    def get_sentence_embeddings(self, texts: List[str], pooling: str = None):
        pass

    def get_aggregated_embeddings(self, texts: List[str]):
        return torch.mean(self.get_sentence_embeddings(texts), dim=0)


class HFTextEmbedder(TextEmbedder):

    def __init__(self, model_name: str = None, device: str = "cpu", pooling: str = "cls", use_auth_token: bool = False,
                 tokenizer: AutoTokenizer = None, model: AutoModel = None, max_length: int = None,
                 inference_batch_size: int = 128, use_cache: bool = False, converted_to_hf: bool = True):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = tokenizer if tokenizer else AutoTokenizer.from_pretrained(model_name, use_auth_token=use_auth_token)
        if model_name in ["Salesforce/SFR-Embedding-Mistral"]:
            self.tokenizer.add_eos_token = True
            self.max_length = 4096
        else:
            self.max_length = max_length
        self.model = model if model else AutoModel.from_pretrained(model_name, use_auth_token=use_auth_token).to(device)
        self.model.eval()

        self.device = device
        self.pooling = pooling
        self.embed_dim = self.model.config.hidden_size

        self.inference_batch_size = inference_batch_size
        self.converted_to_hf = converted_to_hf

        self.use_cache = use_cache
        self.embedding_cache = dict()

        self.logger = logging.getLogger(__name__)

    def get_sentence_embeddings(self, texts: List[str], pooling: str = None):
        self.logger.debug(f"Computing embeddings for {len(texts)} sentences")
        if self.use_cache and str(texts) in self.embedding_cache:
            return self.embedding_cache[str(texts)]
        # make sure items are strings, not None and not NaN
        texts = [str(text) for text in texts if text is not None and text == text]

        if len(texts) == 0:
            return torch.zeros((0, self.embed_dim), dtype=torch.float32, device=self.device)

        if pooling is None:
            pooling = self.pooling

        # Tokenize sentences
        encoded_input = self.tokenizer(texts, max_length=self.max_length if self.max_length is not None else None,
                                       padding=True, truncation=True, return_tensors='pt').to(self.device)

        # Split into batches
        encoded_inputs = []
        for i in range(0, len(encoded_input['input_ids']), self.inference_batch_size):
            encoded_inputs.append({k: v[i:i+self.inference_batch_size] for k, v in encoded_input.items()})

        model_outputs = []
        for batch_encoded_input in encoded_inputs:
            # Compute token embeddings
            with torch.no_grad():
                if self.converted_to_hf:
                    if pooling == "avg_first_last":
                        model_output = self.model(**batch_encoded_input, output_hidden_states=True)
                    else:
                        model_output = self.model(**batch_encoded_input)
                else:
                    if pooling == "avg_first_last":
                        model_output = self.model(**batch_encoded_input, output_hidden_states=True, sent_emb=True)
                    else:
                        model_output = self.model(**batch_encoded_input, sent_emb=True)
            model_outputs.append(model_output)

        # Concatenate all outputs
        if pooling == "avg_first_last":
            model_output = BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=torch.cat([o.last_hidden_state for o in model_outputs]),
                hidden_states=torch.cat([o.hidden_states for o in model_outputs]),
                attentions=torch.cat([o.attentions for o in model_outputs]),
                pooler_output=torch.cat([o.pooler_output for o in model_outputs])
            )
        elif pooling != "last_token":
            model_output = BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=torch.cat([o.last_hidden_state for o in model_outputs]),
                hidden_states=None,
                attentions=None,
                pooler_output=torch.cat([o.pooler_output for o in model_outputs])
            )
        else:
            model_output = BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=torch.cat([o.last_hidden_state for o in model_outputs]),
                hidden_states=None,
                attentions=None,
                pooler_output=None
            )

        # Perform pooling
        if pooling == "avg":
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        elif pooling == "avg_first_last":
            sentence_embeddings = self.mean_pooling_first_last_avg(model_output, encoded_input['attention_mask'])
        elif pooling == "cls_before_pooler":
            sentence_embeddings = self.cls_pooling(model_output)
        elif pooling == "cls":  #cls after mlp
            sentence_embeddings = self.cls_mlp_pooling(model_output)
        elif pooling == "last_token":
            sentence_embeddings = self.last_token_pool(model_output, encoded_input['attention_mask'])
        else:
            raise ValueError("Pooling method not supported")

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        # Cache embeddings
        if self.use_cache:
            self.embedding_cache[str(texts)] = sentence_embeddings
        return sentence_embeddings

    # Mean Pooling - Take attention mask into account for correct averaging
    @staticmethod
    def mean_pooling(model_output: ModelOutput, attention_mask: torch.Tensor):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @staticmethod
    def mean_pooling_first_last_avg(model_output, attention_mask):
        if model_output.hidden_states is None:
            raise Exception(
                "The model output should have hidden_states activated by adding the parameter output_hidden_states=True when retrieving the output!")
        first_layer_embeddings = model_output.hidden_states[
            1]  # [0] is the embedding layer, while [1] should be the first layer
        last_layer_embeddings = model_output.hidden_states[-1]
        avg_first_last_layer_embeddings = torch.mean(torch.stack([first_layer_embeddings, last_layer_embeddings]),
                                                     dim=0)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(avg_first_last_layer_embeddings.size()).float()
        return torch.sum(avg_first_last_layer_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9)

    @staticmethod
    def cls_pooling(model_output):
        return model_output[0][:, 0]

    @staticmethod
    def cls_mlp_pooling(model_output):
        return model_output.pooler_output

    @staticmethod
    def last_token_pool(model_output, attention_mask):
        last_hidden_states = model_output.last_hidden_state
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

