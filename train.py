import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from transformers import TrainerCallback, get_scheduler
import torch.distributed as dist
from typing import Optional
import transformers
import pandas as pd
import polars as pl
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers import AdamW, get_linear_schedule_with_warmup

from models import BertForCL
from data_collator import BCRDataCollator, MyDataCollatorMLM, BCRDatasetForCL
from utils import BCRInsightDataset
from transformers import RobertaTokenizer, BertTokenizer 
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead, RobertaForMaskedLM
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead, BertOnlyMLMHead
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
from transformers import BertConfig, Trainer, TrainingArguments
from datasets import load_dataset
from transformers.data.data_collator import DataCollatorMixin



config = BertConfig(
    vocab_size=260,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=150,
    type_vocab_size=2, 
    initializer_range=0.02,
    layer_norm_eps=1e-12,
    position_embedding_type="absolute",
    
    max_len=145,
    method='sup2',
    p_aa_reverse = -1,
    aas_col = 'bcr_full_aa',
    
    using_v_gene = True,
    using_d_gene = True,
    using_j_gene = True,
    using_c_gene = True,
    
    using_cross_attn=False,         
    lstm_n_layers=1,
    mlp_dropout=0.1,
    init_weight=True
)

# 额外model_args
class ModelArgs:
    pooler_type = "cls"
    temp = 0.05
    do_mlm = True
    mlp_only_train = False
    hard_negative_weight = 0.5
    mlm_weight = 1
    sent_emb = False

model_args = ModelArgs()
device = torch.device("cuda:0")
model = BertForCL(config, config_extra=model_args).to(device)

tokenizer_path = './tokenizer'
data_path = './train_dataset.parquet'


train_df = pl.read_csv('./datasets/train_dataset_001.csv')

# Tokenizer
# tokenizer = BertTokenizer(vocab_file=, merges_file=None)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)


train_dataset = BCRInsightDataset(
    dataframe=train_df, 
    tokenizer=tokenizer,
    max_len=config.max_len,
    method=config.method,
    return_aas_chem = config.using_cross_attn,
    p_aa_reverse=config.p_aa_reverse,
    aas_col_name = config.aas_col, 
    # all_labels = all_labels,
    using_v_gene = config.using_v_gene,
    using_d_gene = config.using_d_gene,
    using_j_gene = config.using_j_gene,
    using_c_gene = config.using_c_gene
)

# Data collator
data_collator = MyDataCollatorMLM(tokenizer=tokenizer, mlm_probability=0.15, padding='longest')
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    # num_train_epochs=1,
    max_steps=225000,
    per_device_train_batch_size=128,
    
    label_names = ['labels', 'mlm_labels'],
    
    gradient_accumulation_steps = 4,
    save_steps=5000,
    save_total_limit=2,
    learning_rate=8e-5,
    dataloader_num_workers = 4, 
    # warmup_steps = 10000,
    warmup_ratio=0.05,
    weight_decay=0.01,
    bf16=True,                      
    deepspeed="ds_config.json",      
    ddp_find_unused_parameters=False,
    eval_strategy="no",
    logging_dir='./logs',
    logging_steps=10,
    logging_first_step = True, 
    report_to="tensorboard"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    data_collator=data_collator,
    # optimizers=(optimizer, scheduler)
)

trainer.train()
# 保存
trainer.save_model('./trained_model')