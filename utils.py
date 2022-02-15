#
# SPDX-FileCopyrightText: 2022 SAP SE or an SAP affiliate company
#
# SPDX-License-Identifier: Apache-2.0

import csv
import os
import copy
import logging
import random
import logging
import pandas as pd
from tqdm import tqdm, trange
from collections import OrderedDict
import re
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, Dataset
from torch.nn import functional as F
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, MaskedLMOutput
from torch.nn import CrossEntropyLoss
from torch import nn
from transformers import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaLMHead, RobertaPreTrainedModel

from transformers.models.bert.modeling_bert import BertForMaskedLM
from scorer import scorer
import os
from tqdm import tqdm, trange
import einops
from tqdm import trange, tqdm
from collections import OrderedDict
from data_reader import InputExample, DataProcessor


ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    'distilroberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
    'roberta-base-openai-detector': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
    'roberta-large-openai-detector': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
}
processor = DataProcessor()

experiment_dict = {'text_original': 0 , 'text_voice': 1, 'text_tense': 2, 'text_number': 3, 'text_gender': 4, 'text_syn': 5, 'text_adverb': 6, 'text_scrambled':7, 'text_freqnoun': 8, 'text_random_context': 9, 'text_rel_clause': 10, 'text_decoy': 11}



class DiscriminatorMultipleChoice(nn.Module):
    def __init__(self, config):
        self.config = config
        super(DiscriminatorMultipleChoice, self).__init__()

        #self.avg_pool = nn.AvgPool2d((8, 10), stride=(6, 8))
        self.pre_cls_stack = nn.Sequential(
            nn.Linear(config.cls_embed_length, config.cls_hidden_size),
            nn.BatchNorm1d(config.cls_hidden_size),
            nn.PReLU(),
            nn.Dropout(config.cls_hidden_dropout),
            #nn.BatchNorm1d(config.cls_hidden_size),
            #nn.PReLU(),
        )
        self.classifier = nn.Linear(
            config.cls_hidden_size, config.cls_num_choices)

    def forward(self, data, labels):
        cls_inpput = self.pre_cls_stack(data)
        logits = self.classifier(cls_inpput)
        # reshaped_logits = logits.view(-1, self.num_choices)
        loss_fct = CrossEntropyLoss()
        multiple_choice_cls_loss = loss_fct(
            logits, labels)

        return multiple_choice_cls_loss



class BertForMaskedLM_MultipleChoice(BertPreTrainedModel):
    
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        
        self.avg_pool = nn.AvgPool2d((8, 10), stride=(6, 8))
        self.pre_cls_stack = nn.Sequential(
            nn.Linear(config.cls_embed_length, config.cls_hidden_size),
            nn.BatchNorm1d(config.cls_hidden_size),
            nn.PReLU(),
            nn.Dropout(config.cls_hidden_dropout)
        )
        self.classifier = nn.Linear(
            config.cls_hidden_size, config.cls_num_choices)
        self.num_choices = config.cls_num_choices

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

  
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        masked_lm_labels=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        multiple_choice_labels = None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        


        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss() 
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            
            
        
        
            
            
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='none')  # -1 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.permute(0,2,1), masked_lm_labels)
            
            masked_lm_loss = torch.div(torch.mean(masked_lm_loss,1),(masked_lm_labels > -1).sum(dim=1,dtype=torch.float32))
            
            masked_lm_loss[torch.isnan(masked_lm_loss)] = 0.0

        

        multiple_choice_lm_loss = None
        if multiple_choice_labels is not None:
            pooled_output = self.avg_pool(torch.hstack(outputs[1]))

            pooled_output = self.pre_cls_stack(
                einops.rearrange(pooled_output, 'a b c -> a (b c)'))
            reshaped_logits = self.classifier(pooled_output)
            loss_fct = CrossEntropyLoss()
            multiple_choice_lm_loss = loss_fct(reshaped_logits, multiple_choice_labels)
            
            

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=[masked_lm_loss] if multiple_choice_lm_loss is None else [
                masked_lm_loss, multiple_choice_lm_loss],
            logits=prediction_scores,
            hidden_states=outputs.hidden_states, #[-2:], # keep only the last
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}
    
    
class RobertaForMaskedLM(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        masked_lm_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='none')  # -1 index = padding token
            #masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            masked_lm_loss = loss_fct(prediction_scores.permute(0,2,1),  masked_lm_labels)
            
            masked_lm_loss = torch.div(torch.mean(masked_lm_loss,1),(masked_lm_labels > -1).sum(dim=1,dtype=torch.float32))
            
            masked_lm_loss[torch.isnan(masked_lm_loss)] = 0.0

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



class RobertaForMaskedLM_old(BertPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForMaskedLM.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForMaskedLM, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                masked_lm_labels=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if masked_lm_labels is not None:
     
           
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='none') 
            
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            
            masked_lm_loss = loss_fct(prediction_scores.permute(0,2,1), masked_lm_labels)
            
            masked_lm_loss_normalized = torch.div(torch.mean(masked_lm_loss,1),(masked_lm_labels > -1).sum(dim=1,dtype=torch.float32))
            
            masked_lm_loss_normalized[torch.isnan(masked_lm_loss_normalized)] = 0.0
            
            outputs = (masked_lm_loss_normalized,) + outputs

        return outputs 
    
class BertForMaskedLM(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

  
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        masked_lm_labels=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='none') 
            masked_lm_loss = loss_fct(prediction_scores.permute(0,2,1), masked_lm_labels)
         
            masked_lm_loss = torch.div(torch.mean(masked_lm_loss,1),(masked_lm_labels > -1).sum(dim=1,dtype=torch.float32))
            
            masked_lm_loss[torch.isnan(masked_lm_loss)] = 0.0
            
            

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states, 
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

# Exponential moving average
class EMA(object):
    def __init__(self, embedding_layer):
        self.embedding_dict_1 = OrderedDict()
        self.embedding_dict_2 = OrderedDict()
        self.embedding_layer = embedding_layer
    def update(self, emb_1, emb_2, guids, alpha):
        
    
        for i in range(len(guids)):
            key = guids[i].item()
            if not(key in self.embedding_dict_1):
                self.embedding_dict_1[key] = emb_1[i,:,:]
                self.embedding_dict_2[key] = emb_2[i,:,:]
            else:
                self.embedding_dict_1[key] = self.embedding_dict_1[key] * (1.-alpha) + emb_1[i,:,:] * alpha
                self.embedding_dict_2[key] = self.embedding_dict_2[key] * (1.-alpha) + emb_2[i,:,:] * alpha
                
                
    def update_from_data(self, ema_data, model, device, alpha, batch_size=5):
        #for i,(_, labels) in enumerate(dataloader.data[1]):
        for i, (train, target,) in enumerate(ema_data):
            
            # re-init list after batch was processed or very first time
            if ( i == 0 or ( i>1 and (i-1) % batch_size == 0)):
            #if True:
                label_input_ids_1 = []
                label_input_ids_2 = []
                label_attention_mask_1 = []
                label_attention_mask_2 = []
                label_segment_ids_1 = []
                label_segment_ids_2 = []
                label_masked_lm_1 = []
                label_masked_lm_2 = []
                label_guids = []
                
            for _, (_, v) in enumerate(target.items()):
            
                #for _,(_,data), in enumerate(labels[0].items()):
                data = v
                
               
                label_input_ids_1.append(data['input_ids_1'].to(device))
                label_input_ids_2.append(data['input_ids_2'].to(device))
                label_attention_mask_1.append(data['attention_mask_1'].to(device))
                label_attention_mask_2.append(data['attention_mask_2'].to(device))
                label_segment_ids_1.append(data['segment_ids_1'].to(device))
                label_segment_ids_2.append(data['segment_ids_2'].to(device))
                label_masked_lm_1.append(data['masked_lm_1'].to(device))
                label_masked_lm_2.append(data['masked_lm_2'].to(device))
                label_guids.append(data['guid'])
                
            # put the batch if it's the last item or accumulation reached batch size
            if ( i == len(ema_data)-1 or (i > 0 and i % batch_size == 0)):
            #if True:
                #print(str(i) + " / "+str(len(ema_data)))
                label_input_ids_1 = torch.stack(label_input_ids_1)
                label_input_ids_2 = torch.stack(label_input_ids_2)
                label_attention_mask_1 = torch.stack(label_attention_mask_1)
                label_attention_mask_2 = torch.stack(label_attention_mask_2)
                label_segment_ids_1 = torch.stack(label_segment_ids_1)
                label_segment_ids_2 = torch.stack(label_segment_ids_2)
                label_masked_lm_1 = torch.stack(label_masked_lm_1)
                label_masked_lm_2 = torch.stack(label_masked_lm_2)
                label_guids = torch.stack(label_guids)
                label_ref_output = []

                if isinstance(model, BertModel) or isinstance(model, RobertaModel):
                    label_ref_output.append(  model.forward(label_input_ids_1, token_type_ids = label_segment_ids_1, attention_mask = label_attention_mask_1).hidden_states[self.embedding_layer].cpu().detach())
                    label_ref_output.append(  model.forward(label_input_ids_2, token_type_ids = label_segment_ids_2, attention_mask = label_attention_mask_2).hidden_states[self.embedding_layer].cpu().detach())
                    
                    self.update(label_ref_output[0].last_hidden_state.cpu().detach(), label_ref_output[1].last_hidden_state.cpu().detach(), label_guids.cpu().detach(), alpha)
                    
                else:
                    label_ref_output.append(  model.forward(label_input_ids_1, token_type_ids = label_segment_ids_1, attention_mask = label_attention_mask_1, masked_lm_labels=label_masked_lm_1, output_hidden_states=True).hidden_states[self.embedding_layer].cpu().detach())
                    label_ref_output.append(  model.forward(label_input_ids_2, token_type_ids = label_segment_ids_2, attention_mask = label_attention_mask_2, masked_lm_labels=label_masked_lm_2, output_hidden_states=True).hidden_states[self.embedding_layer].cpu().detach())
                    
                    self.update(label_ref_output[0], label_ref_output[1], label_guids, alpha)
           
    def get_embeddings(self, guids):
        
        tmp_list_1 = []
        tmp_list_2 = []
        for i in range(len(guids)):
            tmp_list_1.append(self.embedding_dict_1[guids[i].item()])
            tmp_list_2.append(self.embedding_dict_2[guids[i].item()])

        embeddings_1 = torch.stack(tmp_list_1)
        embeddings_2 = torch.stack(tmp_list_2)
        return embeddings_1, embeddings_2
        


class InputExampleX(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, candidate_a, candidate_b, ex_true=True, mex=False, label=None, cat=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. Sentence analysed with pronoun replaced for _
            candidate_a: string, correct candidate
            candidate_b: string, incorrect candidate
        """
        self.guid = guid
        self.text_a = text_a
        self.candidate_a = candidate_a
        self.candidate_b = candidate_b #only used for train
        self.ex_true = ex_true
        self.mex = mex
        self.label = label
        self.cat = cat

def find_sublist(sublist, parent):
    results = []
    sub_len = len(sublist)
    valid_starts = [n for n, word in enumerate(parent) if word == sublist[0]]

    for i in valid_starts:
        if parent[i:i + sub_len] == sublist:
            results.append(i)

    return np.array(results)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids_1, input_ids_2, attention_mask_1, attention_mask_2, type_1, type_2, masked_lm_1, masked_lm_2, start, end_1, end_2, source_start_token_1, source_end_token_1, source_start_token_2, source_end_token_2):
        self.input_ids_1=input_ids_1
        self.attention_mask_1=attention_mask_1
        self.type_1=type_1
        self.masked_lm_1=masked_lm_1
        #These are only used for train examples
        self.input_ids_2=input_ids_2
        self.attention_mask_2=attention_mask_2
        self.type_2=type_2
        self.masked_lm_2=masked_lm_2
        self.start = start
        self.end_1 = end_1
        self.end_2 = end_2
        self.source_start_token_1 = source_start_token_1
        self.source_end_token_1 = source_end_token_1
        self.source_start_token_2 = source_start_token_2
        self.source_end_token_2 = source_end_token_2
        
        



def convert_examples_to_features_trainX(examples, max_seq_len, tokenizer, mode='oxford', logger=None):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    count = [0, 0]
    for (ex_index, example) in enumerate(examples):
        #try:
        if True:
            args = {}
            if isinstance(tokenizer, RobertaTokenizer):
                args = {'add_prefix_space': True}
    
            tokens_sent = tokenizer.tokenize(
                example.text_a.lower(),**args)
            tokens_a = tokenizer.tokenize(
                example.candidate_a.lower(), **args)
            tokens_b = tokenizer.tokenize(
                example.candidate_b.lower(), **args)
            if len(tokens_a) == len(tokens_b):
                count[0] = count[0]+1
            else:
                count[1] = count[1]+1
            tokens_1, type_1, attention_mask_1, masked_lm_1 = [], [], [], []
            tokens_2, type_2, attention_mask_2, masked_lm_2 = [], [], [], []
            tokens_1.append(tokenizer.cls_token)
            tokens_2.append(tokenizer.cls_token)
            try:
                guid = int(example.guid)
            except ValueError:
                #if ex_index == 0:
                #    logger.info("GUIDs are converted to integers")
                guid = ex_index
                pass
            
            if hasattr(example, 'cat') and not(example.cat is None):
               
                tokens_1.append(tokenizer.additional_special_tokens[experiment_dict[example.cat]])
                tokens_2.append(tokenizer.additional_special_tokens[experiment_dict[example.cat]])
                
            for token in tokens_sent:

                if token.find("_") > -1:
                    start = len(tokens_1)
                    if mode == 'oxford':
                        tokens_1.extend(
                            [tokenizer.mask_token for _ in range(len(tokens_a))])
                        tokens_2.extend(
                            [tokenizer.mask_token for _ in range(len(tokens_b))])
                    else:
                        tokens_1.append(tokenizer.mask_token)
                        tokens_2.append(tokenizer.mask_token)

                    end_1 = len(tokens_1)
                    end_2 = len(tokens_2)
                else:
                    tokens_1.append(token)
                    tokens_2.append(token)

     

        token_idx_1 = []
        token_idx_2 = []
        token_counter_1 = 0
        token_counter_2 = 0
        find_tokens_a = True
        find_tokens_b = True

        for idx, token in enumerate(tokens_a):

            if (find_tokens_a and token.lower() == tokens_a[token_counter_1].lower()):
                token_idx_1.append(idx)
                token_counter_1 += 1
                if (len(token_idx_1) >= len(tokens_a)):
                    find_tokens_a = False
            elif find_tokens_a:
                token_idx_1 = []
                token_counter_1 = 0

        for idx, token in enumerate(tokens_b):

            if (find_tokens_b and token.lower() == tokens_b[token_counter_2].lower()):
                token_idx_2.append(idx)
                token_counter_2 += 1
                if (len(token_idx_2) >= len(tokens_b)):
                    find_tokens_b = False
            elif find_tokens_b:
                token_idx_2 = []
                token_counter_2 = 0

        tokens_1 = tokens_1[:max_seq_len-1]  # -1 because of [SEP]
        tokens_2 = tokens_2[:max_seq_len-1]
        if tokens_1[-1] != tokenizer.sep_token:
            tokens_1.append(tokenizer.sep_token)
        if tokens_2[-1] != tokenizer.sep_token:
            tokens_2.append(tokenizer.sep_token)

        type_1 = max_seq_len*[0]  # We do not do any inference.
        type_2 = max_seq_len*[0]  # These embeddings can thus be ignored

        attention_mask_1 = (len(tokens_1)*[1]) + \
            ((max_seq_len-len(tokens_1))*[0])
        attention_mask_2 = (len(tokens_2)*[1]) + \
            ((max_seq_len-len(tokens_2))*[0])

        #sentences
        input_ids_1 = tokenizer.convert_tokens_to_ids(tokens_1)
        input_ids_2 = tokenizer.convert_tokens_to_ids(tokens_2)
        #replacements
        input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)
        input_ids_b = tokenizer.convert_tokens_to_ids(tokens_b)

      
      

       

        for token in tokens_1:
            if token == tokenizer.mask_token:
                if len(input_ids_a) <= 0:
                    continue  # broken case
                masked_lm_1.append(input_ids_a[0])
                input_ids_a = input_ids_a[1:]
            else:
                masked_lm_1.append(-1)
        while len(masked_lm_1) < max_seq_len:
            masked_lm_1.append(-1)

        for token in tokens_2:
            if token == tokenizer.mask_token:
                if len(input_ids_b) <= 0:
                    continue  # broken case
                masked_lm_2.append(input_ids_b[0])
                input_ids_b = input_ids_b[1:]
            else:
                masked_lm_2.append(-1)
        while len(masked_lm_2) < max_seq_len:
            masked_lm_2.append(-1)

        # Zero-pad up to the sequence length.
        while len(input_ids_1) < max_seq_len:
            input_ids_1.append(0)
        while len(input_ids_2) < max_seq_len:
            input_ids_2.append(0)
        assert len(input_ids_1) == max_seq_len
        assert len(input_ids_2) == max_seq_len
        assert len(attention_mask_1) == max_seq_len
        assert len(attention_mask_2) == max_seq_len
        assert len(type_1) == max_seq_len
        assert len(type_2) == max_seq_len
        assert len(masked_lm_1) == max_seq_len
        assert len(masked_lm_2) == max_seq_len
       
        input_ids_1 = torch.tensor(input_ids_1, dtype=torch.long)
        input_ids_2 = torch.tensor(input_ids_2, dtype=torch.long)
        attention_mask_1 = torch.tensor(attention_mask_1, dtype=torch.long)
        attention_mask_2 = torch.tensor(attention_mask_2, dtype=torch.long)
        segment_ids_1 = torch.tensor(type_1, dtype=torch.long)
        segment_ids_2 = torch.tensor(type_2, dtype=torch.long)
        masked_lm_1 = torch.tensor(masked_lm_1, dtype=torch.long)
        masked_lm_2 = torch.tensor(masked_lm_2, dtype=torch.long)
        start = torch.tensor(start, dtype=torch.int16)
        end_1 = torch.tensor(end_1, dtype=torch.int16)
        end_2 = torch.tensor(end_2, dtype=torch.int16)   
        source_start_token_1 = torch.tensor(token_idx_1[0], dtype=torch.int16)
        source_start_token_2 = torch.tensor(token_idx_2[0], dtype=torch.int16)
        source_end_token_1 = torch.tensor(token_idx_1[-1], dtype=torch.int16)
        source_end_token_2 = torch.tensor(token_idx_2[-1], dtype=torch.int16)
        guid = torch.tensor(guid, dtype=torch.int16)
        
        # set the label, or if it is not labeled (target embedding), set it to -1
        if hasattr(example, 'cat') and not(example.cat is None):
            label = experiment_dict[example.cat]
        else:
            label = -1
        
        tmp_dict = dict()
        for variable in ["label", "input_ids_1", "input_ids_2", "attention_mask_1", "attention_mask_2", "segment_ids_1", "segment_ids_2", "masked_lm_1", "masked_lm_2", "start", "end_1", "end_2", "source_start_token_1", "source_start_token_2", "source_end_token_1", "source_end_token_2", "guid"]:
            tmp_dict[variable] = eval(variable)
        features.append(tmp_dict)
    return features



def convert_examples_to_features_evaluate(examples, max_seq_len, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.candidate_a)
        tokens_sent = tokenizer.tokenize(example.text_a)
        
        tokens_1, type_1, attention_mask_1, masked_lm_1 = [],[],[],[]
        tokens_1.append(tokenizer.cls_token)
        for token in tokens_sent:
            if token.find("_")>-1:
                tokens_1.extend([tokenizer.mask_token for _ in range(len(tokens_a))])
            else:
                tokens_1.append(token)
        tokens_1 = tokens_1[:max_seq_len-1]#-1 because of [SEP]
        if tokens_1[-1]!=tokenizer.sep_token:
            tokens_1.append(tokenizer.sep_token)

        type_1 = max_seq_len*[0]
        attention_mask_1 = (len(tokens_1)*[1])+((max_seq_len-len(tokens_1))*[0])
        #sentences
        input_ids_1 = tokenizer.convert_tokens_to_ids(tokens_1)
        #replacements
        input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)

        for token in tokens_1:
            if token==tokenizer.mask_token:
                if len(input_ids_a)<=0:
                    continue#broken case
                masked_lm_1.append(input_ids_a[0])
                input_ids_a = input_ids_a[1:]
            else:
                masked_lm_1.append(-1)
        while len(masked_lm_1)<max_seq_len:
            masked_lm_1.append(-1)
        # Zero-pad up to the sequence length.
        while len(input_ids_1) < max_seq_len:
            input_ids_1.append(0)
        assert len(input_ids_1) == max_seq_len
        assert len(attention_mask_1) == max_seq_len
        assert len(type_1) == max_seq_len
        assert len(masked_lm_1) == max_seq_len

        features.append(
                InputFeatures(input_ids_1=input_ids_1,
                              input_ids_2=None,
                              attention_mask_1=attention_mask_1,
                              attention_mask_2=None,
                              type_1=type_1,
                              type_2=None,
                              masked_lm_1=masked_lm_1,
                              masked_lm_2=None, start=None, end_1=None, end_2=None, source_start_token_1=None, source_end_token_1=None, source_start_token_2=None, source_end_token_2=None))
    return features



class WSCPerturbedDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, tokenizer, csv_file='enhanced_wsc.tsv', root_dir='', perturbations=3, transform=None, shuffle_batch = True, max_seq_length = 100, mode = 'oxford', skip_noncomplete=False, EXPERIMENT_ARR=None, verbose=False, return_orig = False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if perturbations > 0:
            if EXPERIMENT_ARR is None:
                EXPERIMENT_ARR = [('text_original', 'pron_index'),
                      ('text_voice', 'pron_index_voice'),
                      ('text_tense', 'pron_index_tense'),
                      ('text_number', 'pron_index_number'),
                      ('text_gender', 'pron_index'),
                      ('text_syn', 'pron_index_syn'),
                      ('text_adverb', 'pron_index_adverb')]
        else:
            print('No perturbation')
            if EXPERIMENT_ARR is None:
                EXPERIMENT_ARR = [('text_original', 'pron_index')]
            
        
        self.shuffle_batch = shuffle_batch
        self.return_orig = return_orig
        
        wsc_datapoints = pd.read_csv(os.path.join(root_dir,csv_file), sep='\t')
        self.perturbations = perturbations
        data_items = []
        for n, (q_index, entry) in enumerate(wsc_datapoints.iterrows()):
            
           
            if skip_noncomplete:
                found = 0
                for m, (exp_name, pron_col) in enumerate(EXPERIMENT_ARR):
                   
                       
                    if (entry[exp_name].replace(' ', '') in [None, '-']):
                        continue
                    else:
                        found += 1
                if found < perturbations+1:
                    if verbose:
                        print('Skipping element '+str(n))
                  
                    continue
              
            tmp_untagged = None
            if True:
                if self.shuffle_batch:
                    data_items.append(
                        [OrderedDict(), OrderedDict()])#, OrderedDict()])
                    
                for m, (exp_name, pron_col) in enumerate(EXPERIMENT_ARR):
                    current_dict = {}

                    if entry[exp_name].replace(' ', '') in [None, '-']:
                        continue

                    try:
                        text = entry[exp_name].split(" ")
                    except:
                        print(n)
                    text_uncased = [i.lower() for i in text]



                    suffix = "_" + exp_name.split("_")[1] if exp_name in ['text_gender', 'text_number'] else ""
                    pronoun = [entry['pron{}'.format(suffix)].lower()]

                    pron_index = int(entry[pron_col])
                    matched_prons = find_sublist(pronoun, text_uncased)
                    best_match = (np.abs(matched_prons - pron_index)).argmin()
                    pron_index = matched_prons[best_match]

                    text[pron_index] = '_'
                    text = " ".join(text).lower()

                    suffix = "_" + exp_name.split("_")[1] if exp_name in ['text_syn', 'text_gender', 'text_number'] else ""
                    text_option_A = entry['answer_a{}'.format(suffix)]
                    text_option_B = entry['answer_b{}'.format(suffix)]

                    correct_answer = entry['correct_answer'].strip().strip('.').replace(' ', '')
                    correct_answer = str(["a", "b"].index(correct_answer.lower()) + 1)

                    current_dict['sentence'] = text
                    current_dict['option1'] = text_option_A
                    current_dict['option2'] = text_option_B
                    current_dict['answer'] = correct_answer
                    current_dict['pair_number'] = entry['pair_number']
                    current_dict['qID'] = "{}{}".format(m, n)
                    try:
                        current_dict['associative'] = int(entry['associative'])
                        current_dict['switchable'] = int(entry['switchable'])
                    except:
                        current_dict['associative'] = 0
                        current_dict['switchable'] = 0


                    record = current_dict

                    guid = int(record['qID'])
                    sentence = record['sentence']

                    name1 = record['option1']
                    name2 = record['option2']
                    if not 'answer' in record:
                        # This is a dummy label for test prediction.
                        # test.jsonl doesn't include the `answer`.
                        label = "1"
                    else:
                        label = record['answer']

                    conj = "_"
                    idx = sentence.index(conj)
                    context = sentence[:idx]
                    option_str = "_ " + sentence[idx + len(conj):].strip()

                    option1 = option_str.replace("_", name1)
                    option2 = option_str.replace("_", name2)

                    if label == "1":
                        mc_example = InputExampleX(guid,sentence,name1,name2,cat=None)
                    else:
                        mc_example = InputExampleX(guid,sentence,name2,name1,cat=None)
                    #data_items[n].append(current_dict)
                    #data_items[n].append(mc_example)
                    if self.shuffle_batch:
                        if m == 0:
                            tmp_untagged = copy.deepcopy(mc_example)
                            tmp_untagged.cat = exp_name
                            #data_items[-1][2][exp_name] = convert_examples_to_features_trainX([mc_example], max_seq_length, tokenizer, mode)[0]
                            data_items[-1][1][exp_name] = convert_examples_to_features_trainX(
                                [mc_example], max_seq_length, tokenizer, mode)[0]
                            data_items[-1][0][exp_name] = convert_examples_to_features_trainX(
                                [tmp_untagged], max_seq_length, tokenizer, mode)[0]
                        else:
                            # change special tag to 'target label'
                            tmp_untagged.cat = exp_name
                            # we append at the last item because some items might be skipped
                            data_items[-1][1][exp_name] = convert_examples_to_features_trainX([mc_example], max_seq_length, tokenizer, mode)[0]
                            data_items[-1][0][exp_name] = convert_examples_to_features_trainX([tmp_untagged], max_seq_length, tokenizer, mode)[0]
                    else:
                        data_items.append(convert_examples_to_features_trainX([mc_example], max_seq_length, tokenizer, mode)[0])
                    

            self.data = data_items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        if self.shuffle_batch and self.perturbations > 0:
            sidx = random.sample(self.data[idx][0].keys(), np.min([self.perturbations, len(self.data[idx][0])]) )
            #print(sidx)
            if self.return_orig:
                return [self.data[idx][0][j] for j in sidx], [self.data[idx][1][j] for j in sidx], [self.data[idx][1][j] for j in ['text_original']], [experiment_dict[i] for i in sidx]
            else:
                return [self.data[idx][0][j] for j in sidx], [self.data[idx][1][j] for j in sidx]
        else:
            return self.data[idx] #self.data[idx][0]
        
        
def convert_examples_to_features_train(examples, max_seq_len, tokenizer, mode='oxford'):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    count = [0, 0]
    for (ex_index, example) in enumerate(examples):
        try:
            args = {}
            if isinstance(tokenizer, RobertaTokenizer):
                args = {'add_prefix_space': True}
    
            tokens_sent = tokenizer.tokenize(
                example.text_a.lower(),**args)
            tokens_a = tokenizer.tokenize(
                example.candidate_a.lower(), **args)
            tokens_b = tokenizer.tokenize(
                example.candidate_b.lower(), **args)
            if len(tokens_a) == len(tokens_b):
                count[0] = count[0]+1
            else:
                count[1] = count[1]+1
            tokens_1, type_1, attention_mask_1, masked_lm_1 = [], [], [], []
            tokens_2, type_2, attention_mask_2, masked_lm_2 = [], [], [], []
            tokens_1.append(tokenizer.cls_token)
            tokens_2.append(tokenizer.cls_token)
            for token in tokens_sent:

                if token.find("_") > -1:
                    start = len(tokens_1)
                    if mode == 'oxford':
                        tokens_1.extend(
                            [tokenizer.mask_token for _ in range(len(tokens_a))])
                        tokens_2.extend(
                            [tokenizer.mask_token for _ in range(len(tokens_b))])
                    else:
                        tokens_1.append(tokenizer.mask_token)
                        tokens_2.append(tokenizer.mask_token)

                    end_1 = len(tokens_1)
                    end_2 = len(tokens_2)
                else:
                    tokens_1.append(token)
                    tokens_2.append(token)

        except:
            logger.info("Issue with item "+str(ex_index)+"...")
            continue

        token_idx_1 = []
        token_idx_2 = []
        token_counter_1 = 0
        token_counter_2 = 0
        find_tokens_a = True
        find_tokens_b = True

        for idx, token in enumerate(tokens_a):

            if (find_tokens_a and token.lower() == tokens_a[token_counter_1].lower()):
                token_idx_1.append(idx)
                token_counter_1 += 1
                if (len(token_idx_1) >= len(tokens_a)):
                    find_tokens_a = False
            elif find_tokens_a:
                token_idx_1 = []
                token_counter_1 = 0

        for idx, token in enumerate(tokens_b):

            if (find_tokens_b and token.lower() == tokens_b[token_counter_2].lower()):
                token_idx_2.append(idx)
                token_counter_2 += 1
                if (len(token_idx_2) >= len(tokens_b)):
                    find_tokens_b = False
            elif find_tokens_b:
                token_idx_2 = []
                token_counter_2 = 0

        tokens_1 = tokens_1[:max_seq_len-1]  # -1 because of [SEP]
        tokens_2 = tokens_2[:max_seq_len-1]
        if tokens_1[-1] != tokenizer.sep_token:
            tokens_1.append(tokenizer.sep_token)
        if tokens_2[-1] != tokenizer.sep_token:
            tokens_2.append(tokenizer.sep_token)

        type_1 = max_seq_len*[0]  # We do not do any inference.
        type_2 = max_seq_len*[0]  # These embeddings can thus be ignored

        attention_mask_1 = (len(tokens_1)*[1]) + \
            ((max_seq_len-len(tokens_1))*[0])
        attention_mask_2 = (len(tokens_2)*[1]) + \
            ((max_seq_len-len(tokens_2))*[0])

        #sentences
        input_ids_1 = tokenizer.convert_tokens_to_ids(tokens_1)
        input_ids_2 = tokenizer.convert_tokens_to_ids(tokens_2)
        #replacements
        input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)
        input_ids_b = tokenizer.convert_tokens_to_ids(tokens_b)

      
      

       

        for token in tokens_1:
            if token == tokenizer.mask_token:
                if len(input_ids_a) <= 0:
                    continue  # broken case
                masked_lm_1.append(input_ids_a[0])
                input_ids_a = input_ids_a[1:]
            else:
                masked_lm_1.append(-1)
        while len(masked_lm_1) < max_seq_len:
            masked_lm_1.append(-1)

        for token in tokens_2:
            if token == tokenizer.mask_token:
                if len(input_ids_b) <= 0:
                    continue  # broken case
                masked_lm_2.append(input_ids_b[0])
                input_ids_b = input_ids_b[1:]
            else:
                masked_lm_2.append(-1)
        while len(masked_lm_2) < max_seq_len:
            masked_lm_2.append(-1)

        # Zero-pad up to the sequence length.
        while len(input_ids_1) < max_seq_len:
            input_ids_1.append(0)
        while len(input_ids_2) < max_seq_len:
            input_ids_2.append(0)
        assert len(input_ids_1) == max_seq_len
        assert len(input_ids_2) == max_seq_len
        assert len(attention_mask_1) == max_seq_len
        assert len(attention_mask_2) == max_seq_len
        assert len(type_1) == max_seq_len
        assert len(type_2) == max_seq_len
        assert len(masked_lm_1) == max_seq_len
        assert len(masked_lm_2) == max_seq_len
  
        
        input_ids_1 = torch.tensor(input_ids_1, dtype=torch.long)
        input_ids_2 = torch.tensor(input_ids_2, dtype=torch.long)
        attention_mask_1 = torch.tensor(attention_mask_1, dtype=torch.long)
        attention_mask_2 = torch.tensor(attention_mask_2, dtype=torch.long)
        segment_ids_1 = torch.tensor(type_1, dtype=torch.long)
        segment_ids_2 = torch.tensor(type_2, dtype=torch.long)
        masked_lm_1 = torch.tensor(masked_lm_1, dtype=torch.long)
        masked_lm_2 = torch.tensor(masked_lm_2, dtype=torch.long)
        start = torch.tensor(start, dtype=torch.int16)
        end_1 = torch.tensor(end_1, dtype=torch.int16)
        end_2 = torch.tensor(end_2, dtype=torch.int16)   
        source_start_token_1 = torch.tensor(token_idx_1[0], dtype=torch.int16)
        source_start_token_2 = torch.tensor(token_idx_2[0], dtype=torch.int16)
        source_end_token_1 = torch.tensor(token_idx_1[-1], dtype=torch.int16)
        source_end_token_2 = torch.tensor(token_idx_2[-1], dtype=torch.int16)
        
        tmp_dict = dict()
        for variable in ["input_ids_1", "input_ids_2", "attention_mask_1", "attention_mask_2", "segment_ids_1", "segment_ids_2", "masked_lm_1", "masked_lm_2", "start", "end_1", "end_2", "source_start_token_1", "source_start_token_2", "source_end_token_1", "source_end_token_2"]:
            tmp_dict[variable] = eval(variable)
        features.append(tmp_dict)
        
      
    return features


        
def test(processor, args, tokenizer, model, device, global_step = 0, tr_loss = 0, test_set = "wscr-test", verbose=False, output_file=None):
    eval_examples = processor.get_examples(args.data_dir,test_set)
    eval_features = convert_examples_to_features_evaluate(
        eval_examples, args.max_seq_length, tokenizer)
    if verbose:
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids_1 = torch.tensor([f.input_ids_1 for f in eval_features], dtype=torch.long)
    all_attention_mask_1 = torch.tensor([f.attention_mask_1 for f in eval_features], dtype=torch.long)
    all_segment_ids_1 = torch.tensor([f.type_1 for f in eval_features], dtype=torch.long)
    all_masked_lm_1 = torch.tensor([f.masked_lm_1 for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids_1, all_attention_mask_1, all_segment_ids_1, all_masked_lm_1)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    ans_stats=[]
    for batch in eval_dataloader: #tqdm(eval_dataloader,desc="Evaluation"):
        input_ids_1, input_mask_1, segment_ids_1, label_ids_1 = (tens.to(device) for tens in batch)
        with torch.no_grad():
            output = model.forward(input_ids_1, token_type_ids = segment_ids_1, attention_mask = input_mask_1, masked_lm_labels = label_ids_1)

        
        if isinstance(model, BertForMaskedLM_MultipleChoice):
          
            eval_loss = output.loss.to('cpu').numpy()
        else:
            eval_loss = output.loss.to('cpu').numpy()
            
       
        for loss in eval_loss:
            try:
                curr_id = len(ans_stats)
                ans_stats.append((eval_examples[curr_id].guid,eval_examples[curr_id].ex_true,loss))
            except:
                print(curr_id)
                print(len(eval_examples))
                assert False, "error testing"
    if test_set=="gap-test":
        return scorer(ans_stats,test_set,output_file=os.path.join(args.output_dir, "gap-answers.tsv"))
    elif test_set=="wnli":
        return scorer(ans_stats,test_set,output_file=os.path.join(args.output_dir, "WNLI.tsv"))
    else:
        if output_file is not None:
            return scorer(ans_stats,test_set, output_file=os.path.join(args.output_dir, output_file))
        else:
            return scorer(ans_stats,test_set)
        
        
def evaluate(device, model, test_data, embedding_layer):
    model.eval()
    correct = 0
    wrong = 0
    correct_dict = {'text_original': 0, 'text_voice': 0, 'text_tense': 0, 'text_number': 0, 'text_gender': 0, 'text_syn': 0,
                    'text_adverb': 0, 'text_scrambled': 0, 'text_freqnoun': 0, 'text_random_context': 0, 'text_rel_clause': 0}
    wrong_dict = {'text_original': 0, 'text_voice': 0, 'text_tense': 0, 'text_number': 0, 'text_gender': 0, 'text_syn': 0, 'text_adverb': 0, 'text_scrambled': 0,
                  'text_freqnoun': 0, 'text_random_context': 0, 'text_rel_clause': 0}  # {'text_voice': 0, 'text_tense': 0, 'text_number': 0, 'text_gender' : 0, 'text_syn': 0, 'text_adverb': 0}

    for i, (train, target) in enumerate(test_data.data):

        if True:
            input_ids_1 = []
            input_ids_2 = []
            attention_mask_1 = []
            attention_mask_2 = []
            segment_ids_1 = []
            segment_ids_2 = []
            masked_lm_1 = []
            masked_lm_2 = []
            guids = []
            keys = []

        for _, (k, v) in enumerate(train.items()):
            data = v


            input_ids_1.append(data['input_ids_1'].to(device))
            input_ids_2.append(data['input_ids_2'].to(device))
            attention_mask_1.append(data['attention_mask_1'].to(device))
            attention_mask_2.append(data['attention_mask_2'].to(device))
            segment_ids_1.append(data['segment_ids_1'].to(device))
            segment_ids_2.append(data['segment_ids_2'].to(device))
            masked_lm_1.append(data['masked_lm_1'].to(device))
            masked_lm_2.append(data['masked_lm_2'].to(device))
            guids.append(data['guid'])
            keys.append(k)

        for _, (k, v) in enumerate(target.items()):

            data = v


            input_ids_1.append(data['input_ids_1'].to(device))
            input_ids_2.append(data['input_ids_2'].to(device))
            attention_mask_1.append(data['attention_mask_1'].to(device))
            attention_mask_2.append(data['attention_mask_2'].to(device))
            segment_ids_1.append(data['segment_ids_1'].to(device))
            segment_ids_2.append(data['segment_ids_2'].to(device))
            masked_lm_1.append(data['masked_lm_1'].to(device))
            masked_lm_2.append(data['masked_lm_2'].to(device))
            guids.append(data['guid'])
            keys.append(k)


        if True:
            input_ids_1 = torch.stack(input_ids_1)
            input_ids_2 = torch.stack(input_ids_2)
            attention_mask_1 = torch.stack(attention_mask_1)
            attention_mask_2 = torch.stack(attention_mask_2)
            segment_ids_1 = torch.stack(segment_ids_1)
            segment_ids_2 = torch.stack(segment_ids_2)
            masked_lm_1 = torch.stack(masked_lm_1)
            masked_lm_2 = torch.stack(masked_lm_2)
            guids = torch.stack(guids)
            ref_output = []

            if isinstance(model, BertForMaskedLM) or isinstance(model, RobertaForMaskedLM) or isinstance(model, BertForMaskedLM_MultipleChoice) or isinstance(model, RobertaForMaskedLM):
                embedding = model.forward(input_ids_1, token_type_ids = segment_ids_1, attention_mask = attention_mask_1, output_hidden_states=True).hidden_states[embedding_layer]
            else:
                embedding = model.forward(input_ids_1, token_type_ids = segment_ids_1, attention_mask = attention_mask_1).last_hidden_state

            for j in range(len(target.items())):

                per_embedding = einops.repeat(embedding[j,:,:], 'h w -> (b ) h w', b=len(target.items()))#embedding[i][0:len(target.items()),:,:]
                ref_embedding = embedding[len(target.items()):,:,:]
                ref_embedding = ref_embedding.div(torch.norm(ref_embedding, dim=-1).unsqueeze(-1));
                per_embedding = per_embedding.div(torch.norm(per_embedding, dim=-1).unsqueeze(-1));
                tmp = torch.bmm(ref_embedding,per_embedding.transpose(1, 2))

                mask = torch.bmm(attention_mask_1[len(target.items()):,:].unsqueeze(2).float(), attention_mask_1[0:len(target.items()),:].unsqueeze(1).float())
                tmp = tmp * mask
                
                word_precision = tmp.max(dim=2)[0]
                word_recall = tmp.max(dim=1)[0]
                P = (word_precision ).sum(dim=1)
                R = (word_recall ).sum(dim=1)
                F = 2 * P * R / (P + R)
               
                res = torch.argmax(F)
                if res == j:
                    correct += 1
                    correct_dict[keys[j]]=correct_dict[keys[j]]+1
                else:
                    wrong += 1
                    wrong_dict[keys[j]]=wrong_dict[keys[j]]+1

    return ((correct)/(correct+wrong)), {k: (correct_dict[k]/(correct_dict[k]+wrong_dict[k]) if correct_dict[k]+wrong_dict[k] > 0 else 0) for i, (k, v) in enumerate(correct_dict.items())}



def evaluateRanking(args, device, model, tokenizer, test_data, mode, embedding_layer,  topk=3, decoy=None):
    assert mode == 'exact' or mode == 'category', 'Mode has to be either exact or category'
    model.eval()
    
    correct = 0
    wrong = 0
    correct_dict = {'text_original': 0 , 'text_voice': 0, 'text_tense': 0, 'text_number': 0, 'text_gender': 0, 'text_syn': 0, 'text_adverb': 0, 'text_scrambled': 0, 'text_freqnoun': 0, 'text_random_context': 0, 'text_rel_clause': 0}
    wrong_dict = {'text_original': 0, 'text_voice': 0, 'text_tense': 0, 'text_number': 0, 'text_gender': 0, 'text_syn': 0, 'text_adverb': 0, 'text_scrambled': 0,
                  'text_freqnoun': 0, 'text_random_context': 0, 'text_rel_clause': 0}  # {'text_voice': 0, 'text_tense': 0, 'text_number': 0, 'text_gender' : 0, 'text_syn': 0, 'text_adverb': 0}
    
    labels = []
    counter = 0
    for i, (train, target) in enumerate(tqdm(test_data.data)):

        # re-init list after batch was processed or very first time

        if True:
            input_ids_1 = []
            attention_mask_1 = []
            segment_ids_1 = []
    
            guids = []
            keys = []

        for _, (k, v) in enumerate(target.items()):
            data = v


            input_ids_1.append(data['input_ids_1'].to(device))
            attention_mask_1.append(data['attention_mask_1'].to(device))
            segment_ids_1.append(data['segment_ids_1'].to(device))
         
            guids.append(data['guid'])
            keys.append(k)


        # put the batch if it's the last item or accumulation reached batch size
      
        if True:
            input_ids_1 = torch.stack(input_ids_1)
            attention_mask_1 = torch.stack(attention_mask_1)
            segment_ids_1 = torch.stack(segment_ids_1)
            guids = torch.stack(guids)
            ref_output = []

            if isinstance(model, BertForMaskedLM) or isinstance(model, BertForMaskedLM_MultipleChoice) or isinstance(model, RobertaForMaskedLM):
                embedding = model.forward(input_ids_1, token_type_ids = segment_ids_1, attention_mask = attention_mask_1, output_hidden_states=True).hidden_states[embedding_layer]
            else:
                embedding = model.forward(input_ids_1, token_type_ids = segment_ids_1, attention_mask = attention_mask_1).last_hidden_state
            embedding = embedding.div(torch.norm(embedding, dim=-1).unsqueeze(-1)).detach().cpu()
            
            if args.first_feature:
                embedding = embedding[:,0:1,:]
            
            if i == 0:
                all_embeddings = embedding
                all_masks = attention_mask_1.detach().cpu()
            else:
                all_embeddings = torch.cat((all_embeddings, embedding))
                all_masks = torch.cat((all_masks, attention_mask_1.detach().cpu()))
                
            if mode == 'category':
                labels = labels + [i] * embedding.shape[0]
            else:
                for t in range(embedding.shape[0]):
                    counter += 1
                    labels.append(counter)
                    
    if not(decoy is None):
        #print("Elements: "+str(all_embeddings.shape[0]))
        
        train_name = {"gap":"gap-train",
                "wikicrem":"wikicrem-train",
                "dpr":"dpr-train",
                "wscr":"wscr-train",
                "winogrande-xl": "winogrande-xl-train",
                "winogrande-l": "winogrande-l-train",
                "winogrande-m": "winogrande-m-train",
                "winogrande-s": "winogrande-s-train",
                "winogrande-xs": "winogrande-xs-train",
                "all":"all",
                "maskedwiki":"maskedwiki",
                }[decoy]
            
        train_examples = processor.get_examples(args.data_dir, train_name)
        convert_examples_to_features_train(train_examples, args.max_seq_length, tokenizer)
        train_data = convert_examples_to_features_train(train_examples, args.max_seq_length, tokenizer)


        keys.append(k)
            
           
            
        for j in range(0,int(np.ceil(len(train_data)/args.eval_batch_size))):

            data_chunk = train_data[j*args.eval_batch_size:np.min([(j+1)*args.eval_batch_size,len(train_data)])]
            
    
            input_ids_1 = []
            input_ids_2 = []
            attention_mask_1 = []
            attention_mask_2 = []
            segment_ids_1 = []
            segment_ids_2 = []
            masked_lm_1 = []
            masked_lm_2 = []
            for t in range(len(data_chunk)):
                data = data_chunk[t]
                
                
                

                input_ids_1.append(data['input_ids_1'].to(device))
                attention_mask_1.append(data['attention_mask_1'].to(device))
                segment_ids_1.append(data['segment_ids_1'].to(device))
               
                
            input_ids_1 = torch.stack(input_ids_1)
            attention_mask_1 = torch.stack(attention_mask_1)
            segment_ids_1 = torch.stack(segment_ids_1)
           

            if isinstance(model, BertForMaskedLM) or isinstance(model, BertForMaskedLM_MultipleChoice)  or isinstance(model, RobertaForMaskedLM):
                embedding = model.forward(input_ids_1, token_type_ids = segment_ids_1, attention_mask = attention_mask_1, output_hidden_states=True).hidden_states[embedding_layer]
            else:
                embedding = model.forward(input_ids_1, token_type_ids = segment_ids_1, attention_mask = attention_mask_1).last_hidden_state
            embedding = embedding.div(torch.norm(embedding, dim=-1).unsqueeze(-1)).detach().cpu()
            
            if args.first_feature:
                embedding = embedding[:,0:1,:]

            all_embeddings = torch.cat((all_embeddings, embedding))
            all_masks = torch.cat((all_masks, attention_mask_1.detach().cpu()))
            keys = keys + ['text_decoy'] * len(data_chunk)
            labels = labels + [np.max(labels)+1] * len(data_chunk)
        
    correct = 0
    wrong = 0
    counter = 0
    for i, (train, target) in enumerate(tqdm(test_data.data)):
        
        if True:
            input_ids_1 = []
            input_ids_2 = []
            attention_mask_1 = []
            attention_mask_2 = []
            segment_ids_1 = []
            segment_ids_2 = []
            masked_lm_1 = []
            masked_lm_2 = []
            guids = []
            keys = []

        for _, (k, v) in enumerate(train.items()):

            data = v


            input_ids_1.append(data['input_ids_1'].to(device))
          
            attention_mask_1.append(data['attention_mask_1'].to(device))
         
            segment_ids_1.append(data['segment_ids_1'].to(device))
           
            guids.append(data['guid'])
            keys.append(k)
            
        if True:
           
            input_ids_1 = torch.stack(input_ids_1)
           
            attention_mask_1 = torch.stack(attention_mask_1)
          
            segment_ids_1 = torch.stack(segment_ids_1)
           
            guids = torch.stack(guids)
            ref_output = []
            
            if isinstance(model, BertForMaskedLM) or isinstance(model, BertForMaskedLM_MultipleChoice) or isinstance(model, RobertaForMaskedLM):
                embedding = model.forward(input_ids_1, token_type_ids = segment_ids_1, attention_mask = attention_mask_1, output_hidden_states=True).hidden_states[embedding_layer]
            else:
                embedding = model.forward(input_ids_1, token_type_ids = segment_ids_1, attention_mask = attention_mask_1).last_hidden_state
            embedding = embedding.div(torch.norm(embedding, dim=-1).unsqueeze(-1));
            if args.first_feature:
                embedding = embedding[:,0:1,:]
        

        for k in range(0,embedding.shape[0]):
            counter += 1
            for j in range(0,int(np.ceil(all_embeddings.shape[0]/args.eval_batch_size))):
                #print(j)
                input_tensor = all_embeddings[j*args.eval_batch_size:np.min([(j+1)*args.eval_batch_size,all_embeddings.shape[0]]),:,:].to(device)
                tmp = torch.bmm(input_tensor, einops.repeat(embedding[k,:,:], 'h w -> (b ) h w', b=input_tensor.shape[0]).transpose(1, 2))
                
                input_tensor = all_masks[j*args.eval_batch_size:np.min([(j+1)*args.eval_batch_size,all_embeddings.shape[0]]),:].to(device)
                if args.first_feature:
                    mask = torch.bmm(input_tensor[:,0:1].unsqueeze(2).float(), einops.repeat(attention_mask_1[k,0:1], 'h -> (b ) h', b=input_tensor.shape[0]).unsqueeze(1).float())
                else:
                    mask = torch.bmm(input_tensor.unsqueeze(2).float(), einops.repeat(attention_mask_1[k,:], 'h -> (b ) h', b=input_tensor.shape[0]).unsqueeze(1).float())

                tmp = tmp * mask