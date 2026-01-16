from typing import Optional
import random, pandas
import torch, math
import random
import torch.nn as nn
import numpy as np
from scipy.stats import rankdata
from torch.utils.data import random_split, Dataset
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from transformers import TrainerCallback, DataCollatorForLanguageModeling

from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Tuple
from sklearn.metrics import accuracy_score
from datetime import datetime
import logging
import sys, re
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from datetime import datetime
from datetime import timedelta

import functools
import importlib


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def format_time(seconds):
    td = timedelta(seconds=seconds)
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{days} days, {hours:02d}:{minutes:02d}:{seconds:02d}"

def calculate_time_difference(start_time, end_time):
    time_difference = end_time - start_time
    return format_time(time_difference.total_seconds())

def get_log_filename(file):
    current_file_path = os.path.abspath(file)
    file_name = os.path.basename(current_file_path)
    file_name, ext = os.path.splitext(file_name)
    formatted_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return file_name, formatted_time

def get_logger(f_name, model_name, fmt_time):

    log_dir = f'./logs/{model_name}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f'{f_name}_{fmt_time}.log')
    logging.basicConfig(level=logging.INFO,
                        filename=log_file,
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s', 
                        encoding='utf-8')
    logger = logging.getLogger(__name__)
    return logger



# calculation pearson correlation coefficient
def get_pearson_corr(y_true, y_pred):
    fsp = y_pred - torch.mean(y_pred)
    fst = y_true - torch.mean(y_true)
    devP = torch.std(y_pred)
    devT = torch.std(y_true)
    return torch.mean(fsp * fst) / (devP * devT)

# calculation spearman correlation coefficient
def get_spearman_corr(y_true, y_pred):
    y_true, y_pred = torch.tensor(rankdata(y_true)),torch.tensor(rankdata(y_pred))
    return get_pearson_corr(y_true, y_pred)

# converting affinity data to ms data
def from_ic50(ic50, max_ic50=50000.0):
    x = 1.0 - (np.log(np.maximum(ic50, 1e-32)) / np.log(max_ic50))
    return np.minimum(
        1.0,
        np.maximum(0.0, x))

# converting ms data to affinity data
def to_ic50(ms, max_ic50=50000.0):
    x = max_ic50 ** (1-ms)
    return x

# calculate accuracy
accuracy_func = lambda y_pred, y_true, threshold: accuracy_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy() > threshold)

# Initialize model weights
def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.Conv3d):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.ConvTranspose1d):
        torch.nn.init.normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.ConvTranspose3d):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.BatchNorm1d):
        torch.nn.init.normal_(m.weight.data, mean=1, std=0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, mean=1, std=0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.BatchNorm3d):
        torch.nn.init.normal_(m.weight.data, mean=1, std=0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)
    elif isinstance(m, torch.nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)
    elif isinstance(m, torch.nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)
    elif isinstance(m, torch.nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)

# Determine whether to learn masked token based on label
def weight_loss(label, mask_token, mask_label):
    batch_size, max_pred, token_num = mask_label.shape[0], mask_label.shape[1], mask_label.shape[2]
    max_num = (abs(torch.max(mask_label.detach().cpu()))+1)*100
    one_tensor = torch.full_like(mask_label, 1)
    mul_label = label.repeat(token_num, max_pred, 1).transpose(0,2)
    one_tensor -= mul_label
    one_tensor = one_tensor.reshape(-1, token_num)
    mask_token = mask_token.view(-1)
    for index in range(one_tensor.shape[0]):
        one_tensor[index, int(mask_token[index].data)] += 1
    one_tensor = torch.where(one_tensor > 1, 1, 0)*max_num
    return mask_label + one_tensor.reshape(-1, max_pred, token_num)


import re
import os

# Get the absolute path of the current project
main_dir = os.path.dirname(os.path.abspath(__file__))

def read_blosum_aa(path):
    '''
    Read the blosum matrix from the file blosum50.txt
    Args:
        1. path: path to the file blosum50.txt
    Return values:
        1. The blosum50 matrix
    '''
    with open(path,"r") as f: 
        blosums = []
        aa={}
        index = 0
        for line in f:
            blosum_list = []
            line = re.sub("\n","",line)
            # line = line.strip(' ')
            for info in re.split("\s+",line):
                try:
                    blosum_list.append(float(info))
                except:
                    if info not in aa and info.isalpha():
                        aa[info] = index
                        index += 1
            if len(blosum_list) > 0:
                blosums.append(blosum_list)
    assert (len(blosums[0]) == len(i) for i in blosums)   
    return blosums,aa


# Obtain BLOSUMS62 matrix (for protein mapping to numerical matrix) and AA (amino acids with their labels)
# BLOSUMS,AA = read_blosum_aa(main_dir + r"/data/blosum.txt")


import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

        # 确保检查点目录存在
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        
    def __call__(self, val_loss, model, epoch, optimizer, scheduler, sum_step, max_step, start_time, log_file):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimizer, scheduler, sum_step, max_step, start_time, log_file=log_file)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimizer, scheduler, sum_step, max_step, start_time, log_file=log_file)
            self.counter = 0

    def save_checkpoint(self, val_loss, model:nn.Module, epoch:int, optimizer, scheduler, 
                        sum_step:int, max_step:int, start_time, log_file = sys.stdout):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_schedule': scheduler.state_dict(),
            'SUM_step': sum_step,
            'MAX_step': max_step,
            'log_file_name': log_file.name,
            'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            }
        print(f'save_checkpoint...', file=log_file, flush=True)
        torch.save(checkpoint, self.path)
        self.val_loss_min = val_loss


def read_dict(aa_path, v_path, d_path, j_path, label_path, isotype_path = None):
    
    with open(aa_path,'r', encoding='utf-8') as f:
        aa_dict = eval(f.read())
    
    with open(v_path,'r', encoding='utf-8') as f:
        v_dict = f.read().strip('\n').split('\n')
        v_dict = { e:i for i,e in enumerate(v_dict)}
        
    with open(d_path,'r', encoding='utf-8') as f:
        d_dict = f.read().strip('\n').split('\n')
        d_dict = { e:i for i,e in enumerate(d_dict)}
        
    with open(j_path,'r', encoding='utf-8') as f:
        j_dict = f.read().strip('\n').split('\n')
        j_dict = { e:i for i,e in enumerate(j_dict)}
        
    with open(label_path,'r', encoding='utf-8') as f:
        label_dict = f.read().strip('\n').split('\n')
        label_dict = { e:i for i,e in enumerate(label_dict)}
    
    ret_dict = {'aa': aa_dict, 
                'v': v_dict, 
                'd' : d_dict, 
                'j': j_dict, 
                'label': label_dict}
    
    if isotype_path:
        with open(isotype_path,'r', encoding='utf-8') as f:
            isotype_dict = f.read().strip('\n').split('\n')
            isotype_dict = { e:i for i,e in enumerate(isotype_dict)}
        
        ret_dict['isotype'] = isotype_dict

    return ret_dict

# 加载 字典
# BCR 5 class
# encode_dict = read_dict(main_dir + r'/data/dictionary/aa_dict.txt',
#                         main_dir + r'/data/dictionary/mIGHV_functional-5-8_23.tab',
#                         # main_dir + r'/data/dictionary/mIGHV_functional-5.tab',
#                         main_dir + r'/data/dictionary/mIGHD_functional-5.tab',
#                         main_dir + r'/data/dictionary/mIGHJ_functional-5.tab',
#                         main_dir + r'/data/dictionary/mcell_subset_label-5.tab',
#                         main_dir + r'/data/dictionary/oas_isotype_dict.txt')
# # BCR 6 class
encode_dict = read_dict(main_dir + r'/data/dictionary/aa_dict.txt',
                        main_dir + r'/data/dictionary/gse_oas_IGHV_6.tab',
                        main_dir + r'/data/dictionary/gse_oas_IGHD_6.tab',
                        main_dir + r'/data/dictionary/gse_oas_IGHJ_6.tab',
                        main_dir + r'/data/dictionary/gse_oas_BType_6.tab',
                        main_dir + r'/data/dictionary/gse_oas_isotype_6.tab')


# [PAD]  [CLS]    [SEP]   [MASK]    AAseq   v  d  j  isotype  cls
word2idx =  ['[PAD]', '[CLS]', '[SEP]', '[MASK]'] + list(encode_dict['aa'].keys()) + list(encode_dict['v'].keys()) + \
            list(encode_dict['d'].keys()) + list(encode_dict['j'].keys()) + list(encode_dict['isotype'].keys())

word2idx = {word: idx for idx, word in enumerate(word2idx)}

word2idx.update({ word: idx for idx, word in enumerate(encode_dict['label'].keys())})

def padding(ids, n_pads, pad_symb=word2idx['[PAD]']):
    
    return ids.extend([pad_symb for _ in range(n_pads)])

def masking_procedure(cand_pos, input_ids, masked_symb=word2idx['[MASK]'], p_mask = 0.8, p_replace = 0.1):
    masked_pos = []
    masked_tokens = []
    for pos in cand_pos:
        masked_pos.append(pos)
        masked_tokens.append(input_ids[pos])
        if random.random() < p_mask:
            input_ids[pos] = masked_symb
        elif random.random() > (p_mask + p_replace):
            rand_word_idx = random.randint(4, 4 + 19)
            input_ids[pos] = rand_word_idx

    return masked_pos, masked_tokens

def torch_required(func):
    """
    Decorator to ensure PyTorch is available before executing the function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if importlib.util.find_spec("torch") is None:
            raise ImportError("This function requires PyTorch to be installed.")
        return func(*args, **kwargs)
    return wrapper

import pandas as pd

class BCRInsightDataset(Dataset):
    def __init__(self, dataframe : pd.DataFrame, tokenizer, max_len, method:str, mode='train', 
                 return_aas_chem=False, balance_train_dataset = False, p_aa_reverse = -1,
                 aas_col_name = 'cdr3_aa',  v_col_name = 'v', d_col_name = 'd', 
                                            j_col_name = 'j', c_col_name = 'isotype',

                 all_labels = ['Naive-B-Cells', 'Memory-B-Cells', 'Plasma-B-Cells', 
                               'Plasmablast', 'Germinal-Center-B-Cells', 'Immature'],
                 using_v_gene=True, using_d_gene = True, 
                 using_j_gene = True, using_c_gene = True
                 ):
        if mode == 'train':
            dataframe = dataframe.sample(frac=1).reset_index(drop=True)
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.method = method
        self.mode = mode
        self.label_indices = {label: dataframe[dataframe['BType'] == label].index.tolist() for label in dataframe['BType'].unique()}
        self.all_labels = all_labels

        self.balance_train_dataset = balance_train_dataset
        self.p_aa_reverse = p_aa_reverse       
        self.aas_col_name = aas_col_name
        
        self.v_col_name = v_col_name
        self.j_col_name = j_col_name
        self.d_col_name = d_col_name
        self.c_col_name = c_col_name
        
        self.using_v_gene = using_v_gene
        self.using_d_gene = using_d_gene
        self.using_j_gene = using_j_gene
        self.using_c_gene = using_c_gene
        
    
    def __len__(self):
        return len(self.dataframe)

    def get_unsupervised_pair(self, index):
        return index, index

    def get_paired_instance2(self, index):
        current_label = self.dataframe.iloc[index]['BType']
        similar_indices = self.label_indices[current_label]
        # similar_indices.remove(index)
        paired_index = random.choice(similar_indices)
        return index, paired_index
    
    def get_paired_instance3(self, index):
        x, x_p = self.get_paired_instance2(index)
        if self.dataframe.iloc[index]['BType'] == 'Naive-B-Cells':
            neg_label = random.choice(['Memory-B-Cells', 'Immature']) 
        elif self.dataframe.iloc[index]['BType'] == 'Memory-B-Cells':
            neg_label = 'Naive-B-Cells'
        elif self.dataframe.iloc[index]['BType'] == 'Immature':
            neg_label = 'Naive-B-Cells'
        else:
            c_label = self.dataframe.iloc[index]['BType']
            neg_labels = list(self.label_indices.keys())
            neg_labels.remove(c_label)
            neg_label = random.choice(neg_labels)

        neg_index = random.choice(self.label_indices[neg_label])

        return x, x_p, neg_index
    
    def __getitem__(self, index):

        if self.mode == 'train':
            if self.balance_train_dataset:
                random_label = random.choice(self.all_labels)
                index = random.choice(self.label_indices[random_label])
            if self.method == 'unsup':
                idxs = self.get_unsupervised_pair(index)# (x, x)
            elif self.method == 'sup2':
                idxs = self.get_paired_instance2(index) # (x, x_p)
            elif self.method == 'sup3':
                idxs = self.get_paired_instance3(index) # (x, x_p, x_neg) 
            else:
                raise NotImplementedError(f"{self.method} not implemented.")
        else:
            idxs = [index]
            
        encodes = []
        labels = []
        for x in idxs:
            row = self.dataframe.iloc[x]
            aas_seq = list(row[self.aas_col_name])
            rev = (random.random() < self.p_aa_reverse and self.mode == 'train')
            sentence1 = aas_seq[::-1 if rev else 1]
            
            sentence2 = []
            if self.using_v_gene and self.dataframe.iloc[x][self.v_col_name] is not None:
                sentence2.append(self.dataframe.iloc[x][self.v_col_name])
                
            if self.using_d_gene and self.dataframe.iloc[x][self.d_col_name] is not None:
                sentence2.append(self.dataframe.iloc[x][self.d_col_name]) 

            if self.using_j_gene and self.dataframe.iloc[x][self.j_col_name] is not None:
                sentence2.append(self.dataframe.iloc[x][self.j_col_name])
                
            if self.using_c_gene and self.dataframe.iloc[x][self.c_col_name] is not None:
                sentence2.append(self.dataframe.iloc[x][self.c_col_name])
            
            if not sentence2:
                sentence2 = None
                
            encoding1 = self.tokenizer.encode_plus(
                sentence1,
                text_pair=sentence2,
                # add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                # truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
                return_token_type_ids=True 
            )
            encodes.append(encoding1)
            labels.append(self.all_labels.index(self.dataframe.iloc[x]['BType']))
              
        
        if self.mode == 'train':
            data = {
                'input_ids': [ enc['input_ids'].flatten() for enc in encodes ] ,
                'attention_mask': [ enc['attention_mask'].flatten() for enc in encodes ],
                'token_type_ids': [ enc['token_type_ids'].flatten() for enc in encodes ],
                'labels': [torch.tensor(lab, dtype=torch.long) for lab in labels] 
            }
        else:
            data = {
                'input_ids': [ enc['input_ids'].flatten() for enc in encodes ][0] ,
                'attention_mask': [ enc['attention_mask'].flatten() for enc in encodes ][0],
                'token_type_ids': [ enc['token_type_ids'].flatten() for enc in encodes ][0],
                'labels': [torch.tensor(lab, dtype=torch.long) for lab in labels][0] 
            }       
        return data


import polars as pl
import torch
from torch.utils.data import Dataset


@dataclass
class MyDataCollatorMLM:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = 'do_not_pad'
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    # mlm: bool = True
    do_mlm : bool = True
    mlm_probability: float = 0.15 # data_args.mlm_probability
    
    def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids', 'mlm_labels']
        bs = len(features)
        if bs > 0:
            num_sent = len(features[0]['input_ids'])
        else:
            return
        flat_features = []
        for feature in features:
            for i in range(num_sent):
                flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})

        batch = self.tokenizer.pad(
            flat_features,
            padding=self.padding,
            # max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        if self.do_mlm:
            batch["mlm_input_ids"], batch["mlm_labels"] = self.mask_tokens(batch["input_ids"])

        batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0] for k in batch}

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        return batch
    
    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        inputs = inputs.clone()
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

class CustomCallback(TrainerCallback):
    def __init__(self, unit = 'epochs', step = 30):
        self.training_loss = []
        self.eval_loss = []
        self.eval_accuracy = []
        self.unit = unit

    def on_log(self, args, state, control, logs=None, **kwargs):
        
        if state.is_local_process_zero:
            
            if 'loss' in logs:
                self.training_loss.append(logs['loss'])
                
            if  'eval_loss' in logs:
                self.eval_loss.append(logs['eval_loss'])
                
            if 'eval_accuracy' in logs:
                self.eval_accuracy.append(logs['eval_accuracy'])

    def plot_metrics(self):
        
        plt.figure(figsize=(15, 5))
        
        if len(self.training_loss) > 0:
            epochs = range(1, len(self.training_loss) + 1)
            plt.subplot(1, 3, 1)
            plt.plot(epochs, self.training_loss, label='Training Loss')
            plt.xlabel(self.unit)
            plt.ylabel('Loss')
            plt.legend()
        
        if len(self.eval_accuracy) > 0:
            epochs = range(1, len(self.eval_accuracy) + 1)
            plt.subplot(1, 3, 2)
            plt.plot(epochs, self.eval_accuracy, label='Evaluation Accuracy')
            plt.xlabel(self.unit)
            plt.ylabel('Accuracy')
            plt.legend()
        
        if len(self.eval_loss) > 0:
            epochs = range(1, len(self.eval_loss) + 1)
            plt.subplot(1, 3, 3)
            plt.plot(epochs, self.eval_loss, label='Evaluation Loss')
            plt.xlabel(self.unit)
            plt.ylabel('Loss')
            plt.legend()
        

        plt.tight_layout()
        plt.show()

def self_mask_tokens(tokenizer, 
        inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None,mlm_probability=0.15
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    inputs = inputs.clone()
    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels