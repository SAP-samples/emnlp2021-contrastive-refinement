#
# SPDX-FileCopyrightText: 2022 SAP SE or an SAP affiliate company
#
# SPDX-License-Identifier: Apache-2.0
#
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team., 2019 Intelligent Systems Lab, University of Oxford, SAP SE
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler
from utils import WSCPerturbedDataset, EMA, test, evaluate, evaluateRanking, BertForMaskedLM, RobertaForMaskedLM, DiscriminatorMultipleChoice
import einops
from tqdm import trange
import numpy as np
import os
import random
import torch
import sys
import os
from transformers import PYTORCH_PRETRAINED_BERT_CACHE
from transformers import BertTokenizer
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from data_reader import DataProcessor
from transformers import RobertaTokenizer, RobertaConfig
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import argparse
import wandb
import logging
processor = DataProcessor()
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


experiment_dict = {'text_original': 0, 'text_voice': 1, 'text_tense': 2, 'text_number': 3, 'text_gender': 4, 'text_syn': 5,
                   'text_adverb': 6, 'text_scrambled': 7, 'text_freqnoun': 8, 'text_random_context': 9, 'text_rel_clause': 10, 'text_decoy': 11}




# compute the importance of each token representation
# whereby index_vector is the vector containing the indices of the tokens with highest match, and weight vector contains the corresponding scores
def computeTokenImportance(index_vector, weight_vector):

    return_weight_vec = torch.zeros((weight_vector.shape[0],weight_vector.shape[1]))
    
    if index_vector.is_cuda:
        return_weight_vec = return_weight_vec.to('cuda')
        
    
    for j in range(weight_vector.shape[0]):
        for i in range(weight_vector.shape[1]):
            return_weight_vec[j,int(index_vector[j, i])] += weight_vector[j, i]
    return return_weight_vec


# compute the scores (precision, recall, F1) for the dot product
# optionally, restrict the greedy token matching to a band around the matrix diagonal
def computeScore(matrix, band=None):
    if not(band is None): # compute band around diagonal
        neg_val = (1.-1e-5)
        matrix = torch.triu(matrix,diagonal=1)- neg_val*torch.triu(matrix,diagonal=band)+(torch.tril(matrix,diagonal=0)- neg_val*torch.tril(matrix,diagonal=-band))
       
    eps = 0.001    
    word_precision = matrix.max(dim=2)[0]
    word_recall = matrix.max(dim=1)[0]
    P = (word_precision ).sum(dim=1)
    R = (word_recall ).sum(dim=1)
    F = 2 * P * R / (P + R + eps)
    
    return P,R,F


def saveModel(model, tokenizer, output_dir):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(None, os.path.join(output_dir, 'training_args.bin'))
    
def saveClassifier(model, output_dir):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    output_dir = os.path.join(output_dir, "classifier.pt")
    
    torch.save(model.state_dict(), output_dir)
    
def loadClassifier(model, output_dir):
    
    output_dir = os.path.join(output_dir, "classifier.pt")
    
    if os.path.isfile(output_dir):
    
        # Choose whatever GPU device number you want
        model.load_state_dict(torch.load(output_dir,
                              map_location=torch.device('cpu')))
        
        return model
    else:
        return None



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the files for the task.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_experimentname",
                        default=False,
                        action='store_true',
                        help="Whether to use Wandb experiment name as folder")
    parser.add_argument("--output_dir",
                        default="model_output/experiment/",
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")
    
    parser.add_argument("--ranking_steps",
                        type=int,
                        default=10,
                        help="Epochs at which the model should be ranking-wise evaluated")
    parser.add_argument("--eval_steps",
                        type=int,
                        default=5,
                        help="Epochs at which the model should be evaluated")
    parser.add_argument("--ema_update_steps",
                        type=int,
                        default=5,
                        help="Epochs at which the EMA should be updated")
    parser.add_argument("--ema_weight",
                        default=0.001,
                        type=float,
                        help="EMA update factors")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--first_feature",
                        default=False,
                        action='store_true',
                        help="Whether only the first token should be used for classification instead of the whole sequence")
    parser.add_argument("--selected_perturbation",
                        default=False,
                        action='store_true',
                        help="Whether to restrict perturbations to selected perturbation set")
    parser.add_argument("-R_weight",
                        default=1.0,
                        type=float,
                        help="Reconstruction loss weight")
    parser.add_argument("--no_save", action='store_true',
                        help="Use this if models should not stored")
    parser.add_argument('--schedule', type=str, default='linear',
                        choices=['linear', 'cosine'],
                        help='Choice of scheduler')
    parser.add_argument("--C_weight",
                        default=0.5,
                        type=float,
                        help="Contrastive loss weight")
    parser.add_argument("--D_weight",
                        default=0.5,
                        type=float,
                        help="Diversity loss weight")
    parser.add_argument("--matrix_band",
                        default=None,
                        type=int,
                        help="Whether to match on matrix diagonal band")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=None, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs",
                        default=1.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', 
                        type=int, 
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")                       
    parser.add_argument('--load_from_file',
                        type=str,
                        default=None,
                        help="Path to the file with a trained model. Default means bert-model is used. Size must match bert-model.")
    
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    
    parser.add_argument('--gen_perturbations',
                        type=int,
                        default=1,
                        help="Number of updates steps semantic perturbations per instance for generation learning")
    parser.add_argument('--reg_perturbations',
                        type=int,
                        default=4,
                        help="Number of updates steps semantic perturbations per instance for regularizer")
    parser.add_argument("--description",
                        default=None,
                        type=str,
                        required=False,
                        help="Wandb experiment description")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--eval_task_name",
                        default='dpr-test',
                        type=str,
                        required=True,
                        help="The name of evaluation task for model selection.")
    
    parser.add_argument('--shuffle_batch', action='store_true',
                        help="Whether to shuffle elements to avoid potential bias.")
    
    parser.add_argument('--cls_hidden_size',
                        type=int,
                        default=128,
                        help="Multiple choice classifier hidden dimensionality")

    parser.add_argument('--cls_hidden_dropout',
                        type=float,
                        default=0.2,
                        help="dropout for classifier representation")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument('--tags', type=str, default=None,
                    help=('tags for wandb, comma separated'))

    args = parser.parse_args()
        
        
    if not (args.tags is None):
        args.tags = [item for item in args.tags.split(',')]

    if args.description is not None:
        wandb.init(project=args.description, tags=args.tags)
    else:
        wandb.init(project="Commonsense-Pretrain-Perturbation_"+args.task_name.lower(), tags=args.tags)
    
    wandb.config.update(args,  allow_val_change=True)

    wandb.config.update({"Command Line": 'python '+' '.join(sys.argv[0:])})
    
    if not(wandb.run.name is None):
        output_name = wandb.run.name
    else:
        output_name = 'dummy-run'
        
    if args.matrix_band == 0:
        args.matrix_band = None

    
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    #logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))
    
    if args.output_experimentname:
        args.output_dir = os.path.join(args.output_dir, output_name)

    
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    set_seed(args)
    
    
    # Prepare model
    if args.load_from_file is None:
        model_name = args.bert_model
    else:
        model_name = args.load_from_file
        
    if args.bert_model.find('roberta') > -1:
        config = RobertaConfig.from_pretrained(model_name)
        
        config.cls_hidden_size = args.cls_hidden_size
        config.cls_hidden_dropout = args.cls_hidden_dropout
        top_k = 4
        config.cls_embed_length = 2*1024*top_k
        config.cls_num_choices = 10

        config.output_attentions = True
        
        model = RobertaForMaskedLM.from_pretrained(model_name, config = config)
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        
        # pick last layer
        embedding_layer = -1
        
    elif args.bert_model.find('bert') > -1:
        # pick last layer
        embedding_layer = -1
        
        config = BertConfig.from_pretrained(model_name)
        

        
        config.cls_hidden_size = args.cls_hidden_size
        config.cls_hidden_dropout = args.cls_hidden_dropout
        top_k = 4
        config.cls_embed_length = 2*1024*top_k
            
        config.output_attentions = True
            
        model = BertForMaskedLM.from_pretrained(
            model_name, config=config)
        tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Perturbation setup
    EXPERIMENT_ARR = [('text_original', 'pron_index'),  # 0:1
                      ('text_voice', 'pron_index_voice'),  # 1:2
                      ('text_tense', 'pron_index_tense'),  # 2:3
                      ('text_number', 'pron_index_number'),  # 3:4
                      ('text_gender', 'pron_index'),  # 4:5
                      ('text_syn', 'pron_index_syn'),  # 5:6
                      ('text_adverb', 'pron_index_adverb'),  # 6:7
                      ('text_scrambled', 'pron_index_scrambled'),  # 7:8
                      ('text_freqnoun', 'pron_index_freqnoun'),  # 8:9
                      ('text_random_context', 'pron_index_context'),  # 9:10
                      ('text_rel_clause', 'pron_index_rel')  # 10:11
                      ]
    
    if args.selected_perturbation:
        # we just use perturbations that can be predicted with comparably high accuracy
        logger.info('Restrict perturbation to selected perturbation set.')
        tmp_arr = EXPERIMENT_ARR[0:1]+EXPERIMENT_ARR[1:2]+EXPERIMENT_ARR[2:3] + \
            EXPERIMENT_ARR[3:4]+EXPERIMENT_ARR[4:5] + \
            EXPERIMENT_ARR[5:6]+EXPERIMENT_ARR[6:7]
        config.cls_num_choices = 7
            
    else:     
        # we use the entire perturbation set  
        tmp_arr = EXPERIMENT_ARR[0:10]
        config.cls_num_choices = 10
    
    perturbation_keys = [name for (name, param) in tmp_arr]


    # set-up the classifier

    perturbation_classifier = DiscriminatorMultipleChoice(config)
    tmp_cls = None
    
    if args.load_from_file is not None:
        tmp_cls = loadClassifier(perturbation_classifier, args.load_from_file)
        
    if  tmp_cls is  not None:
        tmp_cls.eval()
        perturbation_classifier = tmp_cls
        logger.info('Loaded classifier sucessfully ...')
    perturbation_classifier = perturbation_classifier.to(device)
    # extend vocabulary by special tokens
    
    if args.load_from_file is None:
        logger.info('Extend dictionary with special tokens...')
        special_tokens_dict = {'additional_special_tokens': ['[C0]', '[C1]','[C2]','[C3]','[C4]','[C5]','[C6]','[C7]','[C8]','[C9]','[C10]','[C11]']}
        #special_tokens_dict = {'additional_special_tokens': ['<C0>', '<C1>','<C2>','<C3>','<C4>','<C5>','<C6>','<C7>','<C8>','<C9>','<C10>','<C11>']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
    
    
    model.to(device)       
    
    labels_available = True
    # different perturbations per sample
    wsc_perturbed = WSCPerturbedDataset(tokenizer, perturbations=np.max([args.gen_perturbations,  args.reg_perturbations]), max_seq_length=args.max_seq_length, shuffle_batch=args.shuffle_batch, skip_noncomplete=False,EXPERIMENT_ARR=tmp_arr, return_orig=True)
    # dummy call to create structure
    wsc_test = WSCPerturbedDataset(
        tokenizer, perturbations=6, max_seq_length=args.max_seq_length, shuffle_batch=True, EXPERIMENT_ARR=tmp_arr,)
   
    # create data samples/loaders
    wsc_sampler = RandomSampler(wsc_perturbed)
    wsc_dataloader = DataLoader(wsc_perturbed, sampler=wsc_sampler, batch_size=args.train_batch_size, drop_last=True)
    
    param_optimizer = list(model.named_parameters()) 
    train_data = wsc_perturbed

    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0},
        {'params': [p for n,p in list(perturbation_classifier.named_parameters())], 'lr': 1e-2}
        ]
    t_total = num_train_steps = int(
                len(train_data) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    
    if args.schedule == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
        
    acc,res_dict = evaluate(device, model, test_data=wsc_test, embedding_layer=embedding_layer)
    
    
    # set-up exponential moving average if labels are available for which embeddings can be computed
    if labels_available:
        ema = EMA(embedding_layer=embedding_layer)
        ema.update_from_data(wsc_perturbed.data, model, device, 0., batch_size=5)
    
    # set-up model for training
    model.train()
    perturbation_classifier.train()
    
    max_acc_perturbation = 0
    assoc_acc_commonsense = 0
    max_acc_commonsense = 0
    assoc_acc_perturbation = 0
    max_dpr = 0
    
    num_perturbations = args.gen_perturbations
    
   
    for j in trange(int(args.num_train_epochs), desc="Epoch"):
        avg = []
        for i,(wsc) in enumerate(wsc_dataloader):
            
            if labels_available:
                assert len(wsc) == 4, "WSC should contain data and labels"
                batch, labels, orig_data, perturbation_labels = wsc
                perturbation_labels = torch.stack(perturbation_labels).T
            else:
                batch = wsc
              
            # initialize loss terms
            loss = torch.tensor(0., device=device)
            C = torch.tensor(0., device=device)
            R = torch.tensor(0., device=device)
            D = torch.tensor(0., device=device)
            
                
            embedding_gen_pert = []
            embedding_gen_mask = []
            embedding_sup_pert = []
            embedding_sup_mask = []
            
            # we store the embeddings and mask for the 'text_original' perturbation if required
            original_text_embedding = None
            original_text_mask = None
            
            # save the first perturbation, because that's the embedding of the text_original
            # D computation
            if True:     
                # we just want to compute the embedding of the 'original_text'
                
                if labels_available:
                    # in the training set with groundtruth data, we have labels for all perturbations. here  we pick the one with 'text_original' and treat all other ones as 'perturbations' of that one
                    data = orig_data[0]
                else:
                    # in the perturbed dataset we have access to 'all' generated sentences (because we can create all), also the groundtruth unperturbed
                    data = batch['text_original']

                input_ids_1 = data['input_ids_1'].to(device)
                input_ids_2 = data['input_ids_2'].to(device)
                attention_mask_1 = data['attention_mask_1'].to(device)
                attention_mask_2 = data['attention_mask_2'].to(device)
                segment_ids_1 = data['segment_ids_1'].to(device)
                segment_ids_2 = data['segment_ids_2'].to(device)
                masked_lm_1 = data['masked_lm_1'].to(device)
                masked_lm_2 = data['masked_lm_2'].to(device)

                # we only keep track of the positive examples
                if isinstance(model, BertForMaskedLM)  or isinstance(model, RobertaForMaskedLM):
                    original_text_embedding = model.forward(input_ids_1, token_type_ids=segment_ids_1, attention_mask=attention_mask_1,
                                                            masked_lm_labels=masked_lm_1, output_hidden_states=True).hidden_states[embedding_layer]
                else:
                    original_text_embedding = model.forward(
                        input_ids_1, token_type_ids=segment_ids_1, attention_mask=attention_mask_1).last_hidden_state

                original_text_embedding = original_text_embedding.div(torch.norm(
                    original_text_embedding, dim=-1).unsqueeze(-1))
                original_text_mask = attention_mask_1

            for pertubation_sample in range(num_perturbations): 
                if labels_available:
                    data = labels[pertubation_sample]
                    label_attention_mask_1 = data['attention_mask_1'].to(
                        device)
                    label_attention_mask_2 = data['attention_mask_2'].to(
                        device)
                    label_ref_output = []
                    label_emb_1, label_emb_2 = ema.get_embeddings(data['guid'])
                    label_ref_output.append(label_emb_1.to(device))
                    label_ref_output.append(label_emb_2.to(device))

                    data = batch[pertubation_sample]
                    
                    rnd_keys = [perturbation_keys[i-1]
                                for i in list(data['label'].numpy())]
                    
                    input_ids_1 = data['input_ids_1'].to(device)
                    input_ids_2 = data['input_ids_2'].to(device)
                    attention_mask_1 = data['attention_mask_1'].to(device)
                    attention_mask_2 = data['attention_mask_2'].to(device)
                    segment_ids_1 = data['segment_ids_1'].to(device)
                    segment_ids_2 = data['segment_ids_2'].to(device)
                    masked_lm_1 = data['masked_lm_1'].to(device)
                    masked_lm_2 = data['masked_lm_2'].to(device)
                elif labels_available == False:
                    # get the data according to specific perturbations generated
                    # only sample from perturbations NOT the original text
                    
                    if len(perturbation_keys[1:]) < args.train_batch_size: # sampling with replacement
                        rnd_keys = random.choices(
                            perturbation_keys[1:], k=args.train_batch_size)
                    else: # sampling WITHOUT  replacement
                        rnd_keys = random.sample(
                            perturbation_keys[1:], args.train_batch_size)
                        
                    data_idx = list(range(args.train_batch_size))
                    data = batch
                    
                    input_ids_1 = torch.vstack([data[u]['input_ids_1'][i, :] for (i, u) in zip(data_idx, rnd_keys)]).to(device)
                    input_ids_2 = torch.vstack([data[u]['input_ids_2'][i, :] for (i, u) in zip(data_idx, rnd_keys)]).to(device)
                    attention_mask_1 = torch.vstack([data[u]['attention_mask_1'][i, :] for (
                        i, u) in zip(data_idx, rnd_keys)]).to(device)
                    attention_mask_2 = torch.vstack([data[u]['attention_mask_2'][i, :] for (
                        i, u) in zip(data_idx, rnd_keys)]).to(device)
                    
                    segment_ids_1 = torch.vstack([data[u]['segment_ids_1'][i, :] for (i, u) in zip(data_idx, rnd_keys)]).to(device)
                    segment_ids_2 = torch.vstack([data[u]['segment_ids_2'][i, :] for (i, u) in zip(data_idx, rnd_keys)]).to(device)
                    
                    masked_lm_1 = torch.vstack([batch[u]['masked_lm_1'][i, :] for (
                        i, u) in zip(data_idx, rnd_keys)]).to(device)
                    masked_lm_2 = torch.vstack([batch[u]['masked_lm_2'][i, :] for (i, u) in zip(data_idx, rnd_keys)]).to(device)
               
                perturbed_output = []
            
                POSITIVE_SAMPLE = 0
                
                if isinstance(model, BertForMaskedLM) or isinstance(model, RobertaForMaskedLM):
                    multiple_choice_labels = None
                    
                    perturbed_output.append(model.forward(input_ids_1, token_type_ids=segment_ids_1, attention_mask=attention_mask_1,
                                                          masked_lm_labels=masked_lm_1, output_hidden_states=True))
                    perturbed_output.append(model.forward(input_ids_2, token_type_ids=segment_ids_2,
                                                          attention_mask=attention_mask_2, masked_lm_labels=masked_lm_2, output_hidden_states=True))
                else:
                    perturbed_output.append( model.forward(input_ids_1, token_type_ids = segment_ids_1, attention_mask = attention_mask_1))
                    perturbed_output.append( model.forward(input_ids_2, token_type_ids = segment_ids_2, attention_mask = attention_mask_2))
                
                
                    
                # Diversity loss computation - equation (4)
                if True:
                    
                    # compute the similarity of the text_original with the perturbations
                    
                    if isinstance(model, BertForMaskedLM) or isinstance(model, RobertaForMaskedLM): 
                        perturbation_embedding = perturbed_output[POSITIVE_SAMPLE].hidden_states[embedding_layer]
                    else:
                        perturbation_embedding = perturbed_output[POSITIVE_SAMPLE].last_hidden_state
                        
                    perturbation_embedding = perturbation_embedding.div(torch.norm(
                        perturbation_embedding, dim=-1).unsqueeze(-1))
                    
                    tmp = torch.bmm(perturbation_embedding, original_text_embedding.transpose(1, 2))
                    
                    perturbation_mask = attention_mask_1
                    
                    if args.first_feature:
                        mask = torch.bmm(perturbation_mask[:, 0:1].unsqueeze(2).float(), original_text_mask[:, 0:1].unsqueeze(1).float())
                    else:
                        mask = torch.bmm(perturbation_mask.unsqueeze(2).float(), original_text_mask[:, :].unsqueeze(1).float())
                    
                    tmp = tmp * mask
                    
                    original_max_weights, original_max_indices = tmp.max(dim=2)
                    perturbed_max_weights, perturbed_max_indices = tmp.max(dim=1)
                    
                    original_weight_factor = computeTokenImportance(
                        original_max_indices, original_max_weights)
                    
                    perturbed_weight_factor = computeTokenImportance(
                        perturbed_max_indices, perturbed_max_weights)
                    
                    perturbed_sorted_idx = np.argsort(
                        perturbed_weight_factor.detach().cpu().numpy())
                    
                    original_sorted_idx = np.argsort(
                        original_weight_factor.detach().cpu().numpy())
                    
                    top_k = 4
                    
                    perturbed_sorted_idx = perturbed_sorted_idx[:,-top_k:]
                    original_sorted_idx = original_sorted_idx[:, -top_k:]
                    
                    
                    perturbed_sorted_idx = perturbed_sorted_idx[:,-top_k:]
                    original_sorted_idx = original_sorted_idx[:, -top_k:]

                    # top-k stack
                    
                    weighted_perturbation = torch.vstack([einops.rearrange(
                        perturbation_embedding[i, perturbed_sorted_idx[i,:], :], 'a b -> (a b)') for i in range(tmp.shape[0])])
                    
                    weighted_original = torch.vstack([einops.rearrange(
                        original_text_embedding[i, original_sorted_idx[i, :], :], 'a b-> (a b)') for i in range(tmp.shape[0])])
                 
                    multiple_choice_labels = torch.LongTensor([experiment_dict[i] for i in rnd_keys]).to(device)

                    perturbation_cls_loss = perturbation_classifier.forward(
                        (torch.hstack((weighted_original, weighted_perturbation))).to(device), multiple_choice_labels)
                    D += perturbation_cls_loss
                              
                       

                mask = []
                if True:
                    if labels_available:
                        if not args.first_feature:
                            mask.append(torch.bmm(label_attention_mask_1.unsqueeze(2).float(), attention_mask_1.unsqueeze(1).float()))
                            mask.append(torch.bmm(label_attention_mask_2.unsqueeze(2).float(), attention_mask_2.unsqueeze(1).float()))
                        else:
                            mask.append(torch.bmm(label_attention_mask_1[:,0:1].unsqueeze(2).float(), attention_mask_1[:,0:1].unsqueeze(1).float()))
                            mask.append(torch.bmm(label_attention_mask_2[:,0:1].unsqueeze(2).float(), attention_mask_2[:,0:1].unsqueeze(1).float()))
                    else:
                        # without labels, there are also no masks, therefore we use dummies
                        label_attention_mask_1 = torch.empty(attention_mask_1.shape).to(device)
                        label_attention_mask_2 = torch.empty(attention_mask_2.shape).to(device)

                    
                    # we only iterate over the correct samples, range(2) would correspond to the wrong commonsense answer sentence
                    for t in range(1):


                        if isinstance(model, BertForMaskedLM) or isinstance(model, RobertaForMaskedLM):
                            per_embedding = perturbed_output[POSITIVE_SAMPLE].hidden_states[embedding_layer]
                        else:
                            per_embedding = perturbed_output[POSITIVE_SAMPLE].last_hidden_state
                            
                        # in case there is no label information provided (e.g. domain adaption using terms B,C only)
                        # we use the empty tensor as place-holder
                        
                        if labels_available:
                            ref_embedding = label_ref_output[POSITIVE_SAMPLE]
                        else:
                            ref_embedding = torch.empty(per_embedding.shape).to(device)

                        # BERT-score style classification loss (embedding regression)
                        ref_embedding = ref_embedding.div(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
                        per_embedding = per_embedding.div(torch.norm(per_embedding, dim=-1).unsqueeze(-1))
                        
                        # score is only meaningful is labels available
                        if labels_available:
                            if args.first_feature:
                                tmp = torch.bmm(ref_embedding[:,0:1,:],per_embedding[:,0:1,:].transpose(1, 2))
                            else:
                                tmp = torch.bmm(ref_embedding,per_embedding.transpose(1, 2))
                            _, _, F = computeScore(
                                tmp * mask[POSITIVE_SAMPLE], band=args.matrix_band)
                            R += F.sum()
            
            # Contrastive loss - equation (3) in paper
            if True:
            
                embedding_gen_pert = []
                embedding_gen_mask = []
                embedding_sup_pert = []
                embedding_sup_mask = []
                
                
                if labels_available:
                    assert len(wsc) == 4, "WSC should contain data and labels"
                    batch, labels, orig_data, perturbations_labels = wsc
                    perturbations_labels = torch.stack(perturbations_labels).T
                    num_perturbations = len(labels)
                    #assert num_perturbations > 1, "There should be more than 1 perturbation"
                else:
                    batch = wsc
                    if args.shuffle_batch:
                        num_perturbations = args.reg_perturbations
                    else:
                        num_perturbations = 1
                
                if len(perturbation_keys) < num_perturbations: # do sampling WITH replacement
                    rnd_keys = random.choices(
                        list(range(1, len(perturbation_keys))), k=num_perturbations)
                else: # do sampling WITHOUT replacement
                    rnd_keys = random.sample(
                        list(range(1, len(perturbation_keys))), num_perturbations)

                for k in range(num_perturbations):
                    
                
                    if labels_available and num_perturbations > 1:
                        data = labels[k]
                        label_attention_mask_1 = data['attention_mask_1'].to(device)
                        label_attention_mask_2 = data['attention_mask_2'].to(device)
                        label_ref_output = []
                        label_emb_1, label_emb_2 = ema.get_embeddings(data['guid'])
                        label_ref_output.append(label_emb_1.to(device))
                        label_ref_output.append(label_emb_2.to(device))
                        data = batch[k]
                    elif labels_available == False and num_perturbations > 1:
                        # get the data according to specific perturbations generated
                        data = batch[perturbation_keys[rnd_keys[k]]]
                        
                    input_ids_1 = data['input_ids_1'].to(device)
                    input_ids_2 = data['input_ids_2'].to(device)
                    attention_mask_1 = data['attention_mask_1'].to(device)
                    attention_mask_2 = data['attention_mask_2'].to(device)
                    segment_ids_1 = data['segment_ids_1'].to(device)
                    segment_ids_2 = data['segment_ids_2'].to(device)
                    masked_lm_1 = data['masked_lm_1'].to(device)
                    masked_lm_2 = data['masked_lm_2'].to(device)
                    perturbed_output = []
                    POSITIVE_SAMPLE = 0

                    # check if model is a Masked-LM    
                    if isinstance(model, BertForMaskedLM) or isinstance(model, RobertaForMaskedLM):
                
                        perturbed_output.append(model.forward(input_ids_1, token_type_ids=segment_ids_1, attention_mask=attention_mask_1,
                                                            masked_lm_labels=masked_lm_1, output_hidden_states=True))
                    else:
                        perturbed_output.append( model.forward(input_ids_1, token_type_ids = segment_ids_1, attention_mask = attention_mask_1))
                        
                        if isinstance(model, BertForMaskedLM) or isinstance(model, RobertaForMaskedLM):
                            per_embedding = perturbed_output[POSITIVE_SAMPLE].hidden_states[embedding_layer]
                        else:
                            per_embedding = perturbed_output[POSITIVE_SAMPLE].last_hidden_state

                    if labels_available:
                        ref_embedding = label_ref_output[POSITIVE_SAMPLE]
                    else:
                        ref_embedding = torch.empty(
                            per_embedding.shape).to(device)

                    # BERT-score style classification loss (embedding regression)

                    ref_embedding = ref_embedding.div(
                        torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
                    per_embedding = per_embedding.div(
                        torch.norm(per_embedding, dim=-1).unsqueeze(-1))

                    embedding_gen_pert.append(per_embedding.cpu())
                    embedding_gen_mask.append(attention_mask_1.cpu())
                    embedding_sup_pert.append(ref_embedding.cpu())
                    embedding_sup_mask.append(label_attention_mask_1.cpu())
                    
                    if t == 0:  # only consider positives
                      

                        # compute random distances between generated perturbations
                        idx = (list(range(per_embedding.shape[0])))
                        random.shuffle(idx)
                                
                        embedding_1 = per_embedding
                        mask_1 = attention_mask_1
                        
                        embedding_2 = per_embedding[idx]
                        mask_2 = attention_mask_1[idx]

                        if not args.first_feature:
                            tmp = torch.bmm(
                                embedding_1, embedding_2.transpose(1, 2))
                            mask = (torch.bmm(mask_1.unsqueeze(
                                2).float(), mask_2.unsqueeze(1).float()))
                        else:
                            tmp = torch.bmm(
                                embedding_1[:, 0:1, :], embedding_2[:, 0:1, :].transpose(1, 2))
                            mask = (torch.bmm(mask_1[:, 0:1].unsqueeze(
                                2).float(), mask_2[:, 0:1].unsqueeze(1).float()))
                            

                        _, _, F = computeScore(
                            tmp*mask, band=args.matrix_band)
                        
                        C += F.sum()

                # scale the loss terms                
                
                C = args.C_weight * C
                wandb.log(
                    {'Contrastive Loss': C.cpu().detach().item()})
                loss += C
   
     
                D = args.D_weight * D
                wandb.log(
                    {'Diversity Loss': D.cpu().detach().item()})
                loss += D
            
                R = -args.R_weight*R/(args.train_batch_size*num_perturbations)
                wandb.log(
                    {'Reconstruction Loss': R.cpu().detach().item()})
                loss += R
                
            model.zero_grad()
            perturbation_classifier.zero_grad()
            
            
            wandb.log(
                        {'Multi-Task Loss': loss.cpu().detach().item()})
            loss.backward()#retain_graph=True)
            if not args.max_grad_norm is None:
                torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            if torch.isnan(loss):
                sys.exit()
            
        
        if j>0 and j % args.ema_update_steps == 0 and labels_available:
            if num_perturbations > 1:
                model.eval()
                ema.update_from_data(wsc_perturbed.data, model, device, args.ema_weight, batch_size=3)
                model.train()
        
        if (j>0 and j % args.eval_steps == 0) or (j == int(args.num_train_epochs)-1):
            
            res = test(processor, args, tokenizer, model, device,
                       global_step=1, tr_loss=1, test_set=args.eval_task_name)
            print(args.eval_task_name+"-test: ", res)
            if res >= max_dpr:
                max_dpr = res
                wandb.log({'max-'+args.eval_task_name: res})
            wandb.log({args.eval_task_name: res})
            
            acc, acc_dict = evaluate(device, model, test_data=wsc_test, embedding_layer=embedding_layer)
            logger.info(acc_dict)
        
            
            if acc > max_acc_perturbation:
                max_acc_perturbation = acc
                assoc_acc_commonsense = res
                logger.info("Perturbation Acc: "+str(max_acc_perturbation) +' ('+str(assoc_acc_commonsense)+') commonsense acc.')
                
                output_dir = os.path.join(args.output_dir, "best_perturbation")
                if not args.no_save:
                    saveModel(model, tokenizer, output_dir)
                    saveClassifier(perturbation_classifier, output_dir)
                    with open(os.path.join(output_dir, "best_accuracy.txt"), 'w') as f1_report:
                        f1_report.write("{}".format(max_acc_perturbation))
                wandb.config.update({'best_perturbation_accuracy': max_acc_perturbation},
                                    allow_val_change=True)
                wandb.log({'best_perturbation_accuracy': max_acc_perturbation})
                
            if res > max_acc_commonsense:
                max_acc_commonsense = res
                assoc_acc_perturbation = acc
                logger.info(args.val_task_name+": "+ str(res) +' ('+str(assoc_acc_perturbation)+') perturbation acc.')
                output_dir = os.path.join(args.output_dir, "best_commonsense")
                if not args.no_save:
                    saveModel(model, tokenizer, output_dir)
                    saveClassifier(perturbation_classifier, output_dir)
                    with open(os.path.join(output_dir, "best_accuracy.txt"), 'w') as f1_report:
                        f1_report.write("{}".format(max_acc_commonsense))
                    
                wandb.config.update({'best_commonsense_accuracy': max_acc_commonsense},
                                    allow_val_change=True)
                wandb.log({'best_commonsense_accuracy': max_acc_commonsense})
                
            if j > 0 and j % args.ranking_steps == 0:
                ratio_exact_1, _ = evaluateRanking(args, device, model, tokenizer, test_data=wsc_test, mode='exact', embedding_layer=embedding_layer,
                                                            topk=1, decoy='dpr')
                ratio_exact_5, _ = evaluateRanking(args, device, model, tokenizer, test_data=wsc_test, mode='exact', embedding_layer=embedding_layer,
                                                            topk=5, decoy='dpr')
                ratio_category_1, _ = evaluateRanking(args, device, model, tokenizer, test_data=wsc_test, mode='category', embedding_layer=embedding_layer,
                                                            topk=1, decoy='dpr')
                ratio_category_5, _ = evaluateRanking(args, device, model, tokenizer, test_data=wsc_test, mode='category', embedding_layer=embedding_layer,
                                                            topk=5, decoy='dpr')
                wandb.log({"ranking_exact_top-1": ratio_exact_1, "ranking_exact_top-5": ratio_exact_5, "ranking_category_top-1": ratio_category_1, "ranking_category_top-5": ratio_category_5})
    
            model.train()
            perturbation_classifier.train()
    
    
    acc, _ = evaluate(device, model, test_data=wsc_test,
                      embedding_layer=embedding_layer)
    res = test(processor, args, tokenizer, model, device, global_step=1, tr_loss=1, test_set=args.val_task_name)
    logger.info('Last eval - Perturbatation: '+str(acc)+ ' | Commonsense: '+str(res))
      
    if not args.no_save:
        saveModel(model, tokenizer, args.output_dir)
        saveClassifier(perturbation_classifier, args.output_dir)
        with open(os.path.join(args.output_dir, "best_accuracy.txt"), 'w') as f1_report:
            f1_report.write("Experiment: {}\n".format(output_name))
            f1_report.write("Perturbation Acc.: {}\n".format(acc))
            f1_report.write("Commonsense Acc.: {}\n".format(res))
            
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if True:
            res = test(processor, args, tokenizer, model, device, global_step=1,
                       tr_loss=1, test_set="knowref-test")
            print("Knowref-test: ", res)
            wandb.log({'KnowRef': res})

            res = test(processor, args, tokenizer, model, device,
                       global_step=1, tr_loss=1, test_set="gap-test")
            print("GAP-test: ", res)
            wandb.log({'GAP-test': res})

            res = test(processor, args, tokenizer, model, device,
                       global_step=1, tr_loss=1, test_set="dpr-test")
            print("DPR/WSCR-test: ", res)
            wandb.log({'DPR/WSCR-test': res})

            res = test(processor, args, tokenizer, model, device, global_step=1,
                       tr_loss=1, test_set="wsc",  output_file='wsc-eval.tsv')
            print("WSC: ", res)
            wandb.log({'WSC': res})

        if True:
            res = test(processor, args, tokenizer, model, device, global_step=1,
                       tr_loss=1, test_set="winogender")
            print("WinoGender: ", res)
            wandb.log({'WinoGender': res})

            res = test(processor, args, tokenizer, model, device,
                       global_step=1, tr_loss=1, test_set="pdp")
            print("PDP: ", res)
            wandb.log({'PDP': res})

            res = test(processor, args, tokenizer, model, device, global_step=1,
                       tr_loss=1, test_set="winobias-anti1")
            print("WinoBias Anti Stereotyped Type 1: ", res)
            wandb.log({'WinoBias Anti Stereotyped Type 1': res})

            res = test(processor, args, tokenizer, model, device, global_step=1,
                       tr_loss=1, test_set="winobias-pro1")
            print("WinoBias Pro Stereotyped Type 1: ", res)
            wandb.log({'WinoBias Pro Stereotyped Type 1': res})

            res = test(processor, args, tokenizer, model, device, global_step=1,
                       tr_loss=1, test_set="winobias-anti2")
            print("WinoBias Anti Stereotyped Type 2: ", res)
            wandb.log({'WinoBias Anti Stereotyped Type 2': res})

            res = test(processor, args, tokenizer, model, device, global_step=1,
                       tr_loss=1, test_set="winobias-pro2")
            print("WinoBias Pro Stereotyped Type 2: ", res)
            wandb.log({'WinoBias Pro Stereotyped Type 2': res})

            res = test(processor, args, tokenizer, model, device, global_step=1, tr_loss=1, test_set="winogrande-dev", output_file='winogrande-dev-eval.tsv')
            print("Winogrande (dev): ", res)
            wandb.log({'Winogrande-dev': res})

                    
                    
if __name__ == "__main__":
    main()
        
