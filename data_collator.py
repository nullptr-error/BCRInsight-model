# train_bcr.py
import os
import random
import math
from typing import List, Dict, Any, Optional
from typing import Optional, Union, List, Dict, Tuple
from dataclasses import dataclass
import pandas as pd
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from transformers import (
    BertConfig, BertForMaskedLM, BertTokenizerFast,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)

class BCRDatasetForCL(Dataset):
    def __init__(self, dataframe : pl.DataFrame, tokenizer, max_len, method:str, mode='train', 
                 return_aas_chem=False, balance_train_dataset = False, p_aa_reverse = -1,
                 aas_col_name = 'cdr3_aa', v_col_name = 'v_call', d_col_name = 'd_call', j_col_name = 'j_call', c_col_name = 'isotype',
                 all_labels = None, using_v_gene=True, using_d_gene = True, using_j_gene = True, using_c_gene = True):
        
        if all_labels is None:
            all_labels = dataframe['BType'].unique().sort().to_list() 
            #['Naive-B-Cells', 'Memory-B-Cells', 'Plasma-B-Cells', 'Plasmablast', 'Germlinal-Center-B-Cells', 'Immature']
        
        if mode == 'train':
            dataframe = dataframe.sample(fraction=1, shuffle = True, seed = 2025) #.reset_index(drop=True)

        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.method = method
        self.mode = mode
        
        dataframe = dataframe.with_row_count("idx")
        print('+'*50)
        # build label indices
        # self.label_indices = {label: dataframe[dataframe['BType'] == label].index.tolist() for label in dataframe['BType'].unique()}
        
        self.label_indices = {
            label: dataframe.filter(pl.col("BType") == label)["idx"].to_list()
            for label in dataframe["BType"].unique()
        }

        groups = dataframe.group_by(["BType", "v", "j"], maintain_order=True)
        self.vj_indices = {
            f"{btype}_{v}_{j}": group["idx"].to_list()
            for (btype, v, j), group in groups
        }

        # grouped = dataframe.groupby(['BType', 'v', 'j'], sort=False)
        # self.vj_indices = {f"{btype}_{v}_{j}": group.index.tolist() 
        #            for (btype, v, j), group in grouped}
        
        self.all_labels = all_labels
        self.labels = list(self.label_indices.keys())
        self.return_aas_chem = return_aas_chem
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
        return self.dataframe.height

    def get_unsupervised_pair(self, index):
        return index, index

    def get_paired_instance2(self, index):
        
        random_label = random.choice(self.labels)
        indices = self.label_indices[random_label]
        index = random.choice(indices)
                
        current_label = self.dataframe["BType"][index]
        current_v = self.dataframe["v"][index]
        current_j = self.dataframe["j"][index]        
                
        vj_key = f"{current_label}_{current_v}_{current_j}"
        if vj_key in self.vj_indices:
            candidates = [i for i in self.vj_indices[vj_key] if i != index]
            if candidates:
                paired_index = random.choice(candidates)
                return index, paired_index

        candidates = [i for i in self.label_indices[current_label] if i != index]
        paired_index = random.choice(candidates)
        return index, paired_index
    
    def get_paired_instance3(self, index):
        x, x_p = self.get_paired_instance2(index)
        # choose negative label (simple heuristics)

        current = self.dataframe["BType"][index]
        
        possible = list(self.label_indices.keys())
        if current in possible:
            possible.remove(current)
        neg_label = random.choice(possible) if possible else current
        neg_index = random.choice(self.label_indices[neg_label])
        return x, x_p, neg_index

    def __getitem__(self, index):
        if self.mode == 'train':
            # if self.balance_train_dataset:
            #     random_label = random.choice(self.all_labels)
            #     index = random.choice(self.label_indices[random_label])
            if self.method == 'unsup':
                idxs = self.get_unsupervised_pair(index)
            elif self.method == 'sup2':
                idxs = self.get_paired_instance2(index)
            elif self.method == 'sup3':
                idxs = self.get_paired_instance3(index)
            else:
                raise NotImplementedError(f"{self.method} not implemented.")
        else:
            idxs = [index]

        encodes = []
        labels = []
        for x in idxs:
            # row = self.dataframe.iloc[x]
            row = self.dataframe.row(index, named=True)
            aas_seq = list(str(row[self.aas_col_name]))  # split aa string into list of chars (AA tokens)
            # rev = (random.random() < self.p_aa_reverse and self.mode == 'train') if (self.p_aa_reverse is not None and self.p_aa_reverse > 0) else False
            # if rev:
            #     aas_seq = aas_seq[::-1]
            # sentence1: AA sequence as list
            # we will join with space so tokenizer can split tokens if you use token-level tokens in vocab
            # If your tokenizer expects a sequence of tokens rather than characters, adapt accordingly.
            sentence1 = aas_seq

            # build gene tokens (V/D/J/C)
            sentence2_tokens = []
            if self.using_v_gene and pd.notnull(row[self.v_col_name]):
                sentence2_tokens.append(str(row[self.v_col_name]))
            if self.using_d_gene and pd.notnull(row[self.d_col_name]):
                sentence2_tokens.append(str(row[self.d_col_name]))
            if self.using_j_gene and pd.notnull(row[self.j_col_name]):
                sentence2_tokens.append(str(row[self.j_col_name]))
            if self.using_c_gene and pd.notnull(row[self.c_col_name]) and row[self.c_col_name] != 'Bulk':
                sentence2_tokens.append(str(row[self.c_col_name]))

            if not sentence2_tokens:  # 如果 sentence2 是空列表
                sentence2_tokens = None

            encoding1 = self.tokenizer.encode_plus(
                sentence1,
                text_pair=sentence2_tokens,
                truncation=True,
                max_length=self.max_len,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
                return_token_type_ids=True
            )
            encodes.append(encoding1)
            labels.append(self.all_labels.index(row['BType']) if row['BType'] in self.all_labels else -1)

        # return structure: encodes: list of dicts
        # We'll return raw encodings and labels; collator will combine them.
        data = {
            'input_ids': [ enc['input_ids'].flatten() for enc in encodes ] ,                # [tensor, tensor]
            'attention_mask': [ enc['attention_mask'].flatten() for enc in encodes ],       
            'token_type_ids': [ enc['token_type_ids'].flatten() for enc in encodes ],
            'labels': [torch.tensor(lab, dtype=torch.long) for lab in labels] ,             # [tensor, tensor]
            'mode': self.mode,
            'method': self.method            
        }
        return data

# ---------------------------

# ---------------------------
class BCRDataCollator:
    def __init__(self, tokenizer: BertTokenizerFast, mlm_probability: float = 0.15, max_len=145):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.max_len = max_len
        # we can optionally use DataCollatorForLanguageModeling to perform MLM
        self.mlm_collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=mlm_probability)

    def __call__(self, batch):
        # =============================
        # =============================
        sent1 = [x["bcr_full_aa"] for x in batch]
        sent2 = [f"{x['v_call']} {x['d_call']} {x['j_call']} {x['c_call']}" for x in batch]

        # Tokenize 
        main = self.tokenizer(
            sent1,
            sent2,
            truncation=True,
            max_length=145,
            padding=True,
            return_tensors="pt",
        )

        # =============================
        # =============================
        idxs = list(range(len(batch)))
        random.shuffle(idxs)

        sent1_pair = [batch[i]["bcr_full_aa"] for i in idxs]
        sent2_pair = [f"{batch[i]['v_call']} {batch[i]['d_call']} {batch[i]['j_call']} {batch[i]['c_call']}"
                      for i in idxs]

        pair = self.tokenizer(
            sent1_pair,
            sent2_pair,
            truncation=True,
            max_length=145,
            padding=True,
            return_tensors="pt",
        )

        # =============================
        # 3. MLM mask
        # =============================
        main_mlm = self.mlm_collator([{
            "input_ids": ids,
            "attention_mask": am
        } for ids, am in zip(main["input_ids"], main["attention_mask"])])

        pair_mlm = self.mlm_collator([{
            "input_ids": ids,
            "attention_mask": am
        } for ids, am in zip(pair["input_ids"], pair["attention_mask"])])

        return {
            # Main
            "input_ids": main_mlm["input_ids"],
            "attention_mask": main_mlm["attention_mask"],
            "labels": main_mlm["labels"],

            # Positive pair
            "input_ids_pair": pair_mlm["input_ids"],
            "attention_mask_pair": pair_mlm["attention_mask"],
            "labels_pair": pair_mlm["labels"],
        }


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
