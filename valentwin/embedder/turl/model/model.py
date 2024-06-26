from valentwin.embedder.turl.model.transformers.modeling_bert import *


class TableHybridEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(TableHybridEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0, sparse=False)
        self.ent_embeddings = nn.Embedding(config.ent_vocab_size, config.hidden_size, padding_idx=0, sparse=False)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.ent_mask_embedding = nn.Embedding(4, config.hidden_size, padding_idx=0)

        self.fusion = nn.Linear(2 * config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def load_pretrained(self, checkpoint, is_bert=True):
        state_dict = self.state_dict()
        if is_bert:
            state_dict['LayerNorm.weight'] = checkpoint['bert.embeddings.LayerNorm.weight']
            state_dict['LayerNorm.bias'] = checkpoint['bert.embeddings.LayerNorm.bias']
            state_dict['word_embeddings.weight'] = checkpoint['bert.embeddings.word_embeddings.weight']
            state_dict['position_embeddings.weight'] = checkpoint['bert.embeddings.position_embeddings.weight']
            new_type_size = state_dict['type_embeddings.weight'].shape[0]
            state_dict['type_embeddings.weight'] = checkpoint['bert.embeddings.token_type_embeddings.weight'][0].repeat(
                new_type_size).view(new_type_size, -1)
        else:
            for key in state_dict:
                state_dict[key] = checkpoint['table.embeddings.' + key]
        self.load_state_dict(state_dict)

    def forward(self, input_tok=None, input_tok_type=None, input_tok_pos=None, input_ent_tok=None,
                input_ent_tok_length=None, input_ent_mask_type=None, input_ent=None, input_ent_type=None,
                ent_candidates=None):
        tok_embeddings = None
        if input_tok is not None:
            input_tok_embeds = self.word_embeddings(input_tok)
            input_tok_pos_embeds = self.position_embeddings(input_tok_pos)
            input_tok_type_embeds = self.type_embeddings(input_tok_type)

            tok_embeddings = input_tok_embeds + input_tok_pos_embeds + input_tok_type_embeds
            tok_embeddings = self.LayerNorm(tok_embeddings)
            tok_embeddings = self.dropout(tok_embeddings)

        if input_ent is None and input_ent_tok is None:
            return tok_embeddings, None, None

        ent_embeddings = None
        if input_ent_tok is not None:
            input_ent_tok_embeds = self.word_embeddings(input_ent_tok)
            input_ent_tok_embeds = input_ent_tok_embeds.sum(dim=-2)
            input_ent_tok_embeds = input_ent_tok_embeds / input_ent_tok_length[:, :, None]
            if input_ent_mask_type is not None:
                input_ent_mask_embeds = self.ent_mask_embedding(input_ent_mask_type)
                input_ent_tok_embeds = torch.where((input_ent_mask_type != 0)[:, :, None], input_ent_mask_embeds,
                                                   input_ent_tok_embeds)
        if input_ent is not None:
            # if input_ent.is_cuda:
            #     input_ent_embeds = self.ent_embeddings(input_ent.cpu()).cuda()
            # else:
            input_ent_embeds = self.ent_embeddings(input_ent)
            if input_ent_tok is None:
                input_ent_tok_embeds = torch.zeros_like(input_ent_embeds)
        else:
            input_ent_embeds = torch.zeros_like(input_ent_tok_embeds)
        ent_embeddings = self.fusion(torch.cat([input_ent_embeds, input_ent_tok_embeds], dim=-1))
        ent_embeddings = self.transform_act_fn(ent_embeddings)
        ent_embeddings = self.LayerNorm(ent_embeddings)
        ent_embeddings = self.dropout(ent_embeddings)

        if input_ent_type is not None:
            input_ent_type_embeds = self.type_embeddings(input_ent_type)
            ent_embeddings += input_ent_type_embeds

        if ent_embeddings is not None:
            ent_embeddings = self.LayerNorm(ent_embeddings)
            ent_embeddings = self.dropout(ent_embeddings)

        if ent_candidates is not None:
            ent_candidates_embeddings = self.ent_embeddings(ent_candidates)
        else:
            ent_candidates_embeddings = None

        return tok_embeddings, ent_embeddings, ent_candidates_embeddings


class TableLayerSimple(nn.Module):
    def __init__(self, config):
        super(TableLayerSimple, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, tok_hidden_states=None, tok_attention_mask=None, ent_hidden_states=None, ent_attention_mask=None):
        tok_outputs, ent_outputs = (None, None), (None, None)
        if tok_hidden_states is not None:
            if ent_hidden_states is not None:
                tok_self_attention_outputs = self.attention(tok_hidden_states, encoder_hidden_states=torch.cat(
                    [tok_hidden_states, ent_hidden_states], dim=1), encoder_attention_mask=tok_attention_mask)
            else:
                tok_self_attention_outputs = self.attention(tok_hidden_states, encoder_hidden_states=tok_hidden_states,
                                                            encoder_attention_mask=tok_attention_mask)
            tok_attention_output = tok_self_attention_outputs[0]
            tok_outputs = tok_self_attention_outputs[1:]
            tok_intermediate_output = self.intermediate(tok_attention_output)
            tok_layer_output = self.output(tok_intermediate_output, tok_attention_output)
            tok_outputs = (tok_layer_output,) + tok_outputs

        if ent_hidden_states is not None:
            if tok_hidden_states is not None:
                ent_self_attention_outputs = self.attention(ent_hidden_states, encoder_hidden_states=torch.cat(
                    [tok_hidden_states, ent_hidden_states], dim=1), encoder_attention_mask=ent_attention_mask)
            else:
                ent_self_attention_outputs = self.attention(ent_hidden_states, encoder_hidden_states=ent_hidden_states,
                                                            encoder_attention_mask=ent_attention_mask)
            ent_attention_output = ent_self_attention_outputs[0]
            ent_outputs = ent_self_attention_outputs[1:]
            ent_intermediate_output = self.intermediate(ent_attention_output)
            ent_layer_output = self.output(ent_intermediate_output, ent_attention_output)
            ent_outputs = (ent_layer_output,) + ent_outputs

        return tok_outputs, ent_outputs


class TableEncoderSimple(nn.Module):
    def __init__(self, config):
        super(TableEncoderSimple, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([TableLayerSimple(config) for _ in range(config.num_hidden_layers)])

    def load_pretrained(self, checkpoint, is_bert=True):
        state_dict = self.state_dict()
        if is_bert:
            for x in state_dict:
                state_dict[x] = checkpoint['bert.encoder.'+x]
                print('load %s <- %s'%(x, 'bert.encoder.'+x))
        else:
            for x in state_dict:
                state_dict[x] = checkpoint['table.encoder.'+x]
        self.load_state_dict(state_dict)

    def forward(self, tok_hidden_states=None, tok_attention_mask=None, ent_hidden_states=None, ent_attention_mask=None):
        tok_all_hidden_states = ()
        tok_all_attentions = ()
        ent_all_hidden_states = ()
        ent_all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                tok_all_hidden_states = tok_all_hidden_states + (tok_hidden_states,)
                ent_all_hidden_states = ent_all_hidden_states + (ent_hidden_states,)

            tok_layer_outputs, ent_layer_outputs = layer_module(tok_hidden_states, tok_attention_mask, ent_hidden_states, ent_attention_mask)
            tok_hidden_states = tok_layer_outputs[0]
            ent_hidden_states = ent_layer_outputs[0]

            if self.output_attentions:
                tok_all_attentions = tok_all_attentions + (tok_layer_outputs[1],)
                ent_all_attentions = ent_all_attentions + (ent_layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            tok_all_hidden_states = tok_all_hidden_states + (tok_hidden_states,)
            ent_all_hidden_states = ent_all_hidden_states + (ent_hidden_states,)

        tok_outputs = (tok_hidden_states,)
        ent_outputs = (ent_hidden_states,)
        if self.output_hidden_states:
            tok_outputs = tok_outputs + (tok_all_hidden_states,)
            ent_outputs = ent_outputs + (ent_all_hidden_states,)
        if self.output_attentions:
            tok_outputs = tok_outputs + (tok_all_attentions,)
            ent_outputs = ent_outputs + (ent_all_attentions,)
        return tok_outputs, ent_outputs  # last-layer hidden state, (all hidden states), (all attentions)


class TableLayer(nn.Module):
    def __init__(self, config):
        super(TableLayer, self).__init__()
        self.tok_attention = BertAttention(config)
        self.tok_intermediate = BertIntermediate(config)
        self.tok_output = BertOutput(config)
        self.ent_attention = BertAttention(config)
        self.ent_intermediate = BertIntermediate(config)
        self.ent_output = BertOutput(config)

    def forward(self, tok_hidden_states, tok_attention_mask, ent_hidden_states, ent_attention_mask):
        tok_self_attention_outputs = self.tok_attention(tok_hidden_states, encoder_hidden_states=torch.cat(
            [tok_hidden_states, ent_hidden_states], dim=1), encoder_attention_mask=tok_attention_mask)
        tok_attention_output = tok_self_attention_outputs[0]
        tok_outputs = tok_self_attention_outputs[1:]
        tok_intermediate_output = self.tok_intermediate(tok_attention_output)
        tok_layer_output = self.tok_output(tok_intermediate_output, tok_attention_output)
        tok_outputs = (tok_layer_output,) + tok_outputs

        ent_self_attention_outputs = self.ent_attention(ent_hidden_states, encoder_hidden_states=torch.cat(
            [tok_hidden_states, ent_hidden_states], dim=1), encoder_attention_mask=ent_attention_mask)
        ent_attention_output = ent_self_attention_outputs[0]
        ent_outputs = ent_self_attention_outputs[1:]
        ent_intermediate_output = self.ent_intermediate(ent_attention_output)
        ent_layer_output = self.ent_output(ent_intermediate_output, ent_attention_output)
        ent_outputs = (ent_layer_output,) + ent_outputs

        return tok_outputs, ent_outputs


class TableEncoder(nn.Module):
    def __init__(self, config):
        super(TableEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([TableLayer(config) for _ in range(config.num_hidden_layers)])

    def load_pretrained(self, checkpoint):
        raise NotImplementedError

    def forward(self, tok_hidden_states, tok_attention_mask, ent_hidden_states, ent_attention_mask):
        tok_all_hidden_states = ()
        tok_all_attentions = ()
        ent_all_hidden_states = ()
        ent_all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                tok_all_hidden_states = tok_all_hidden_states + (tok_hidden_states,)
                ent_all_hidden_states = ent_all_hidden_states + (ent_hidden_states,)

            tok_layer_outputs, ent_layer_outputs = layer_module(tok_hidden_states, tok_attention_mask, ent_hidden_states, ent_attention_mask)
            tok_hidden_states = tok_layer_outputs[0]
            ent_hidden_states = ent_layer_outputs[0]

            if self.output_attentions:
                tok_all_attentions = tok_all_attentions + (tok_layer_outputs[1],)
                ent_all_attentions = ent_all_attentions + (ent_layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            tok_all_hidden_states = tok_all_hidden_states + (tok_hidden_states,)
            ent_all_hidden_states = ent_all_hidden_states + (ent_hidden_states,)

        tok_outputs = (tok_hidden_states,)
        ent_outputs = (ent_hidden_states,)
        if self.output_hidden_states:
            tok_outputs = tok_outputs + (tok_all_hidden_states,)
            ent_outputs = ent_outputs + (ent_all_hidden_states,)
        if self.output_attentions:
            tok_outputs = tok_outputs + (tok_all_attentions,)
            ent_outputs = ent_outputs + (ent_all_attentions,)
        return tok_outputs, ent_outputs  # last-layer hidden state, (all hidden states), (all attentions)


class HybridTableModel(BertPreTrainedModel):
    def __init__(self, config, is_simple=False):
        super(HybridTableModel, self).__init__(config)
        self.is_simple = is_simple
        self.config = config

        self.embeddings = TableHybridEmbeddings(config)
        if is_simple:
            self.encoder = TableEncoderSimple(config)
        else:
            self.encoder = TableEncoder(config)

        self.init_weights()

    def load_pretrained(self, checkpoint, is_bert=True):
        self.embeddings.load_pretrained(checkpoint, is_bert=is_bert)
        self.encoder.load_pretrained(checkpoint, is_bert=is_bert)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, word_embedding_matrix, ent_embedding_matrix):
        assert self.embeddings.word_embeddings.weight.shape == word_embedding_matrix.shape
        assert self.embeddings.ent_embeddings.weight.shape == ent_embedding_matrix.shape
        self.embeddings.word_embeddings.weight.data = word_embedding_matrix
        self.embeddings.ent_embeddings.weight.data = ent_embedding_matrix

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_tok = None, input_tok_type = None, input_tok_pos = None, input_tok_mask = None,
                input_ent_tok = None, input_ent_tok_length = None, input_ent_mask_type = None,
                input_ent = None, input_ent_type = None, input_ent_mask = None, ent_candidates = None):
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_input_tok_mask, extended_input_ent_mask = None, None
        if input_tok_mask is not None:
            extended_input_tok_mask = input_tok_mask[:, None, :, :]
            extended_input_tok_mask = extended_input_tok_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_input_tok_mask = (1.0 - extended_input_tok_mask) * -10000.0
        if input_ent_mask is not None:
            extended_input_ent_mask = input_ent_mask[:, None, :, :]
            extended_input_ent_mask = extended_input_ent_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_input_ent_mask = (1.0 - extended_input_ent_mask) * -10000.0

        tok_embedding_output, ent_embedding_output, ent_candidates_embeddings = self.embeddings(input_tok, input_tok_type, input_tok_pos, input_ent_tok, input_ent_tok_length, input_ent_mask_type, input_ent, input_ent_type, ent_candidates) #disgard ent_pos since they are all 0
        tok_encoder_outputs, ent_encoder_outputs = self.encoder(tok_embedding_output, extended_input_tok_mask, ent_embedding_output, extended_input_ent_mask)
        tok_sequence_output = tok_encoder_outputs[0]
        ent_sequence_output = ent_encoder_outputs[0]

        tok_outputs = (tok_sequence_output, ) + tok_encoder_outputs[1:]  # add hidden_states and attentions if they are here
        ent_outputs = (ent_sequence_output, ) + ent_encoder_outputs[1:]
        return tok_outputs, ent_outputs, ent_candidates_embeddings  # sequence_output, (hidden_states), (attentions)


class TableLMSubPredictionHead(nn.Module):
    """
    only make prediction for a subset of candidates
    """
    def __init__(self, config, output_dim=None, use_bias=True):
        super(TableLMSubPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config, output_dim=output_dim)
        if use_bias:
            self.bias = nn.Embedding.from_pretrained(torch.zeros(config.ent_vocab_size, 1), freeze=False)
        else:
            self.bias = None

    def forward(self, hidden_states, candidates, candidates_embeddings, return_hidden=False):
        hidden_states = self.transform(hidden_states)
        scores = torch.matmul(hidden_states, torch.transpose(candidates_embeddings,1,2))
        if self.bias is not None:
            scores += torch.transpose(self.bias(candidates),1,2)
        if return_hidden:
            return (scores,hidden_states)
        else:
            return scores


class TableMLMHead(nn.Module):
    def __init__(self, config):
        super(TableMLMHead, self).__init__()
        self.tok_predictions = BertLMPredictionHead(config)
        self.ent_predictions = TableLMSubPredictionHead(config)

    def load_pretrained(self, checkpoint):
        state_dict = self.state_dict()
        for x in state_dict:
            if x.find('tok_predictions') != -1:
                state_dict[x] = checkpoint['cls.' + x[4:]]
                print('load %s <- %s' % (x, 'cls.' + x[4:]))
            elif x.find('bias') == -1:
                state_dict[x] = checkpoint['cls.' + x[4:]]
                print('load %s <- %s' % (x, 'cls.' + x[4:]))
        self.load_state_dict(state_dict)

    def forward(self, tok_sequence_output, ent_sequence_output, ent_candidates, ent_candidates_embeddings):
        tok_prediction_scores = self.tok_predictions(tok_sequence_output)
        ent_prediction_scores = self.ent_predictions(ent_sequence_output, ent_candidates, ent_candidates_embeddings)
        return tok_prediction_scores, ent_prediction_scores


class HybridTableMaskedLM(BertPreTrainedModel):
    def __init__(self, config, is_simple=False):
        super(HybridTableMaskedLM, self).__init__(config)

        self.table = HybridTableModel(config, is_simple)
        self.cls = TableMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.tok_predictions.decoder

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        tok_output_embeddings = self.get_output_embeddings()
        tok_input_embeddings = self.table.get_input_embeddings()
        if tok_output_embeddings is not None:
            self._tie_or_clone_weights(tok_output_embeddings, tok_input_embeddings)

    def load_pretrained(self, checkpoint):
        self.table.load_pretrained(checkpoint)
        self.cls.load_pretrained(checkpoint)

    def forward(self, input_tok, input_tok_type, input_tok_pos, input_tok_mask,
                input_ent_tok, input_ent_tok_length, input_ent_mask_type,
                input_ent, input_ent_type, input_ent_mask, ent_candidates,
                tok_masked_lm_labels=None, ent_masked_lm_labels=None, exclusive_ent_mask=None):
        tok_outputs, ent_outputs, ent_candidates_embeddings = self.table(input_tok, input_tok_type, input_tok_pos,
                                                                         input_tok_mask, input_ent_tok,
                                                                         input_ent_tok_length, input_ent_mask_type,
                                                                         input_ent, input_ent_type, input_ent_mask,
                                                                         ent_candidates)

        tok_sequence_output = tok_outputs[0]
        ent_sequence_output = ent_outputs[0]
        tok_prediction_scores, ent_prediction_scores = self.cls(tok_sequence_output, ent_sequence_output,
                                                                ent_candidates, ent_candidates_embeddings)

        tok_outputs = (tok_prediction_scores,) + tok_outputs  # Add hidden states and attention if they are here
        ent_outputs = (ent_prediction_scores,) + ent_outputs

        # Although this may seem awkward, BertForMaskedLM supports two scenarios:
        # 1. If a tensor that contains the indices of masked labels is provided,
        #    the cross-entropy is the MLM cross-entropy that measures the likelihood
        #    of predictions for masked words.
        # 2. If `lm_labels` is provided we are in a causal scenario where we
        #    try to predict the next token for each input in the decoder.
        # pdb.set_trace()
        if tok_masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            tok_masked_lm_loss = loss_fct(tok_prediction_scores.view(-1, self.config.vocab_size),
                                          tok_masked_lm_labels.view(-1))
            tok_outputs = (tok_masked_lm_loss,) + tok_outputs
        if ent_masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            if exclusive_ent_mask is not None:
                ent_prediction_scores.scatter_add_(2, exclusive_ent_mask,
                                                   (1.0 - (exclusive_ent_mask >= 1000).float()) * -10000.0)
            ent_prediction_scores += (ent_candidates[:, None, :] == 0).float() * -10000.0
            ent_masked_lm_loss = loss_fct(ent_prediction_scores.view(-1, self.config.max_entity_candidate),
                                          ent_masked_lm_labels.view(-1))
            ent_outputs = (ent_masked_lm_loss,) + ent_outputs
        # pdb.set_trace()
        return tok_outputs, ent_outputs  # (masked_lm_loss), (ltr_lm_loss), prediction_scores, (hidden_states), (attentions)