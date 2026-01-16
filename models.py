import torch,os
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from transformers import TrainerCallback, get_scheduler
import torch.distributed as dist
from typing import Optional
import transformers
from transformers import RobertaTokenizer
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

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        # pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    if config.init_weight:
        cls.init_weights()
    cls.config = config


def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
    aa_seq_chem = None,
    chem_attention_mask = None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)    
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    # Separate representation
    z1, z2 = pooler_output[:,0], pooler_output[:,1]

    # Hard negative
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    # Hard negative
    if num_sent >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()

    # Calculate loss with hard negatives
    if num_sent == 3:
        # Note that weights are actually logits of weights
        z3_weight = cls.model_args.hard_negative_weight
        weights = torch.tensor(
            [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
        ).to(cls.device)
        cos_sim = cos_sim + weights

    loss = loss_fct(cos_sim, labels)

    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        attentions=outputs.attentions,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        # self.model_args = model_kargs["model_args"]
        self.model_args = model_kargs["config_extra"]
        
        self.bert = BertModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)
        # if config.using_cross_attn:
        #     self.key_value_lstm_projection = torch.nn.LSTM(
        #             input_size = 553,
        #             hidden_size = config.hidden_size,
        #             num_layers = config.lstm_n_layers,
        #             bias=True,
        #             batch_first=True,
        #             dropout=config.mlp_dropout,
        #             bidirectional=True,
        #             proj_size=config.hidden_size//2
        #         )
        #     self.cross_attn = CrossAttention(hidden_size=config.hidden_size)
        
        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
        
        aas_seq_chem = None,
        chem_attention_mask = None,
    ):
        if self.model_args.sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                
                aa_seq_chem = aas_seq_chem,
                chem_attention_mask = chem_attention_mask,
            )


class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["config_extra"]
        
        if model_kargs.get('model_name', None) is not None:
            self.roberta = RobertaModel.from_pretrained(model_kargs['model_name'], 
                                                        config=config)
        else:
            self.roberta = RobertaModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if self.model_args.sent_emb: 
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )


class RobertaForCLSeqClassification(RobertaForCL):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config, *model_args, **model_kargs)
        
        self.num_labels = config.num_labels # model_kargs['num_labels']
        
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 256),         
            nn.ReLU(),                                
            nn.Dropout(config.mlp_dropout),  
            
            nn.Linear(256, 64),                        
            nn.ReLU(),
            nn.Dropout(config.mlp_dropout),  
            
            nn.Linear(64, self.num_labels)              
        )

        for param in self.roberta.encoder.layer[:].parameters():
            param.requires_grad = False


        for param in self.roberta.encoder.layer[config.unfreeze_n_layers:].parameters():
            param.requires_grad = True
        
    def forward(self, 
                input_ids=None, 
                attention_mask=None, 
                token_type_ids=None, 
                position_ids=None, 
                head_mask=None, 
                inputs_embeds=None, 
                cls_labels=None,
                labels = None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                mlm_input_ids=None,
                mlm_labels=None):
        
        outputs = super().forward(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            position_ids = position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            sent_emb = True,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mlm_input_ids=mlm_input_ids,
            mlm_labels=mlm_labels
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0, :])  # 使用 [CLS] token 进行分类

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads = 12, dropout=0.1):
        super(CrossAttention, self).__init__()

        # MultiheadAttention layer
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size,
                                               num_heads=num_attention_heads,
                                               dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, query, key, value, attention_mask=None):
        """
        :param query: (batch_size, query_length, hidden_size) -> Text features
        :param key: (batch_size, key_length, hidden_size) -> Image features
        :param value: (batch_size, value_length, hidden_size) -> Image features
        :param attention_mask: (batch_size, 1, 1, key_length), optional mask for attention
        :return: (batch_size, query_length, hidden_size)
        """

        # Query, Key, Value shape needs to be [seq_length, batch_size, hidden_size] for multi-head attention
        query = query.transpose(0, 1)  # [query_length, batch_size, hidden_size]
        key = key.transpose(0, 1)      # [key_length, batch_size, hidden_size]
        value = value.transpose(0, 1)  # [value_length, batch_size, hidden_size]

        # Perform attention
        attn_output, attn_output_weights = self.attention(query, key, value, key_padding_mask=attention_mask)

        # attn_output is of shape [query_length, batch_size, hidden_size]
        # We need to transpose it back to [batch_size, query_length, hidden_size]
        attn_output = attn_output.transpose(0, 1)
        query = query.transpose(0, 1)
        
        # Add residual connection and layer normalization
        output = self.layer_norm(query + self.dropout(attn_output))

        return output, attn_output_weights


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout=0.1):
        super(BidirectionalCrossAttention, self).__init__()

        # MultiheadAttention layer
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size,
                                               num_heads=num_attention_heads,
                                               dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, query, key, value, attention_mask=None):
        """
        :param query: (batch_size, query_length, hidden_size) -> Text features
        :param key: (batch_size, key_length, hidden_size) -> Image features
        :param value: (batch_size, value_length, hidden_size) -> Image features
        :param attention_mask: (batch_size, 1, 1, key_length), optional mask for attention
        :return: (batch_size, query_length, hidden_size)
        """

        # Query, Key, Value shape needs to be [seq_length, batch_size, hidden_size] for multi-head attention
        query = query.transpose(0, 1)  # [query_length, batch_size, hidden_size]
        key = key.transpose(0, 1)      # [key_length, batch_size, hidden_size]
        value = value.transpose(0, 1)  # [value_length, batch_size, hidden_size]

        # Perform attention
        attn_output, attn_output_weights = self.attention(query, key, value, key_padding_mask=attention_mask)

        # attn_output is of shape [query_length, batch_size, hidden_size]
        # We need to transpose it back to [batch_size, query_length, hidden_size]
        attn_output = attn_output.transpose(0, 1)

        # Add residual connection and layer normalization
        output = self.layer_norm(query + self.dropout(attn_output))

        return output, attn_output_weights

    def bidirectional_attention(self, text_query, image_query, text_key, image_key, text_value, image_value, attention_mask=None):
        """
        :param text_query: Text query (e.g., text features)
        :param image_query: Image query (e.g., image features)
        :param text_key: Text key (e.g., text features)
        :param image_key: Image key (e.g., image features)
        :param text_value: Text value (e.g., text features)
        :param image_value: Image value (e.g., image features)
        :param attention_mask: Mask for attention
        :return: Updated text and image features after bidirectional cross-attention
        """
        
        # First, apply cross-attention with text as query and image as key/value
        text_to_image, _ = self.forward(text_query, image_key, image_value, attention_mask)
        
        # Second, apply cross-attention with image as query and text as key/value
        image_to_text, _ = self.forward(image_query, text_key, text_value, attention_mask)

        return text_to_image, image_to_text


import torch
import torch.nn as nn
import torch.nn.functional as F


class BertForCLSeqClassification(BertForCL):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config, *model_args, **model_kargs)
        
        self.num_labels = config.num_labels # model_kargs['num_labels']
        self.using_cross_attn = config.using_cross_attn
        
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 256),          # 第一层，输入大小为 input_size，输出大小为 input_size // 2
            nn.ReLU(),                                  # 激活函数
            nn.Dropout(config.mlp_dropout),  
            
            nn.Linear(256, 64),                         # 第二层
            nn.ReLU(),
            nn.Dropout(config.mlp_dropout),  
            
            nn.Linear(64, self.num_labels)              # 第三层，输出大小为 num_labels
        )
        # 冻结全部
        for param in self.bert.encoder.layer[:].parameters():
            param.requires_grad = False

        # 解冻后 n 层
        for param in self.bert.encoder.layer[config.unfreeze_n_layers:].parameters():
            param.requires_grad = True
        
        if config.using_cross_attn:
            # self.key_value_projection = nn.Linear(553, config.hidden_size)
            self.key_value_lstm_projection = torch.nn.LSTM(
                    input_size = 553,
                    hidden_size = config.hidden_size,
                    num_layers = config.lstm_n_layers,
                    bias=True,
                    batch_first=True,
                    dropout=config.mlp_dropout,
                    bidirectional=True,
                    proj_size=config.hidden_size//2
                )
            self.cross_attn = CrossAttention(hidden_size=config.hidden_size)
        
    def forward(self, 
                input_ids=None, 
                attention_mask=None, 
                token_type_ids=None, 
                position_ids=None, 
                head_mask=None, 
                inputs_embeds=None, 
                aa_seq_chem=None,
                chem_attention_mask=None,
                labels = None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                mlm_input_ids=None,
                mlm_labels=None):
        
        outputs = super().forward(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            position_ids = position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            sent_emb = True,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mlm_input_ids=mlm_input_ids,
            mlm_labels=mlm_labels
        )

        sequence_output = outputs[0]   # self.key_value_projection(aa_seq_chem)
        if self.using_cross_attn:
            output, (h_n, c_n) = self.key_value_lstm_projection(aa_seq_chem) 
            sequence_output, attn_output_weights = self.cross_attn(query=sequence_output, 
                                                                   key=output, 
                                                                   value=output, 
                                                                   attention_mask=chem_attention_mask)

        logits = self.classifier(sequence_output[:, 0, :])  # 使用 [CLS] token 进行分类  这里好像不需要啊，已经池化了之前

        loss = None
        if labels is not None:
            
            class_counts = torch.bincount(labels, minlength=self.num_labels)           # 统计每个类别的样本数
            total_samples = len(labels)                     # 样本总数

            # 计算每个类别的权重：频率较低的类别权重大，频率较高的类别权重小
            # 使用倒数的方法：权重 = 总样本数 / (类别数量 * 类别样本数)
            class_weights = total_samples / (len(class_counts) * class_counts.float())

            # 如果某些类别在训练集中没有样本（即样本数为 0），我们为它们设置一个非常小的权重（比如 0.1）
            # 我们用 0.1 来表示类别没有出现在训练集中，或者在训练过程中不太关注这些类别
            class_weights[class_counts == 0] = 1  # 为没有样本的类别设置较小的权重
            class_weights = class_weights.to(logits.device)  # 移动到同样的设备上

            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size=768, attn_type="standard", num_heads=8, hidden_dim=256):
        super(AttentionPooling, self).__init__()
        self.attn_type = attn_type
        self.hidden_size = hidden_size

        if attn_type == "standard":
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, 512),
                nn.Tanh(),
                nn.Linear(512, 256),
                nn.Tanh(),
                nn.Linear(256, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )
        elif attn_type == "self":
            self.query = nn.Linear(hidden_size, hidden_size)
            self.key = nn.Linear(hidden_size, hidden_size)
            self.value = nn.Linear(hidden_size, hidden_size)
            self.scale = hidden_size ** 0.5
        elif attn_type == "multihead":
            self.multi_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads)
        elif attn_type == "gated":
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            self.gate = nn.Sequential(
                nn.Linear(hidden_size, hidden_dim),
                nn.Sigmoid()
            )
        else:
            raise ValueError("Unsupported attn_type. Choose from: standard, self, multihead, gated")

    def forward(self, instance_embeddings):
        # 输入: [1, num_instances, hidden_size] 或 [batch_size, num_instances, hidden_size]
        if self.attn_type == "standard":
            attn_scores = self.attention(instance_embeddings)  # [batch_size * num_instances, 1]attn_scores # 
            attn_weights = F.softmax(attn_scores, dim=0)
            bag_embedding = torch.sum(instance_embeddings * attn_weights, dim=0)  # [batch_size, hidden_size]

        elif self.attn_type == "self":
            q = self.query(instance_embeddings)
            k = self.key(instance_embeddings)
            v = self.value(instance_embeddings)
            attn_scores = torch.softmax(q @ k.transpose(-2, -1) / self.scale, dim=-1)  # [batch_size, num_instances, num_instances]
            context = attn_scores @ v  # [batch_size, num_instances, hidden_size]
            attn_weights = attn_scores.mean(dim=1)  # [batch_size, num_instances]
            bag_embedding = torch.sum(context * attn_weights.unsqueeze(-1), dim=1)  # [batch_size, hidden_size]

        elif self.attn_type == "multihead":
            attn_output, attn_weights = self.multi_attention(instance_embeddings, instance_embeddings, instance_embeddings)
            attn_scores = attn_weights.mean(dim=1)  # [batch_size, num_instances]
            attn_scores = torch.softmax(attn_scores, dim=-1)
            bag_embedding = torch.sum(attn_output * attn_scores.unsqueeze(-1), dim=1)  # [batch_size, hidden_size]

        elif self.attn_type == "gated":
            attn_scores = self.attention(instance_embeddings)   # [batch_size, num_instances, 1]
            gate_scores = self.gate(instance_embeddings)        # [batch_size, num_instances, hidden_dim]
            attn_weights = F.softmax(attn_scores * gate_scores.mean(dim=-1, keepdim=True), dim=0)
            bag_embedding = torch.sum(instance_embeddings * attn_weights, dim=0)  # [batch_size, hidden_size]

        return bag_embedding, attn_weights


class RobertaForMaskedLMSeqClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.roberta = RobertaForMaskedLM(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.num_labels = config.num_labels
        self.mlm_weight = config.mlm_weight
        self.pooler = Pooler(config.pooler_type)
        self.mlp = MLPLayer(config=config)  if config.pooler_type == 'cls' else None

        # self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, labels=None, cls_labels=None):
        
        outputs = self.roberta.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            
            output_hidden_states=True,
            return_dict=True,
            
            labels=labels,  
            
            output_attentions = True
        )
        
        if outputs.hidden_states is not None:
            outputs.last_hidden_state = outputs.hidden_states[-1]
        
        pool_out = self.pooler(attention_mask, outputs)  
        
        if self.mlp is not None:
            pool_out = self.mlp(pool_out)
            
        # sequence_output = outputs.hidden_states
        logits = self.classifier(pool_out)

        loss = outputs.loss
        if cls_labels is not None:
            cls_labels = cls_labels.to(logits.device)
            loss_fct = nn.CrossEntropyLoss()
            classification_loss = loss_fct(logits.view(-1, self.num_labels), cls_labels.view(-1))
            if loss is not None:
                loss = self.mlm_weight * loss + (1 - self.mlm_weight) * classification_loss
            else:
                loss = classification_loss 

        # del outputs.last_hidden_state
        # print(type(loss), loss.shape)
        return SequenceClassifierOutput(
            loss=loss.unsqueeze(0),
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



from transformers import TrainerCallback

class SaveBestModelCallback(TrainerCallback):
    def __init__(self):
        self.best_metric = float('inf')

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_metric = metrics['eval_loss']
        if current_metric < self.best_metric:
            self.best_metric = current_metric
            # 手动保存模型
            control.should_save = True  # 触发保存
        else:
            control.should_save = False  # 不保存

class LogLossToFileCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file_name = log_file
        # 创建文件并写入表头
        self.log_file = open(self.log_file_name, 'w')
        self.log_file.write("step,loss\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        # 如果日志中包含 'loss'，则将损失值记录到文件
        if "loss" in logs:
            step = state.global_step
            # loss = logs["loss"]
            self.log_file.write(f"{step},{str(logs)}\n")
            self.log_file.flush()

class TrainingEarlyStoppingCallback(TrainerCallback):
    def __init__(self, num_warmup_steps, max_steps, optim = None, save_best_model_dir='', 
                 patience_stop=3, threshold_imporv=0.0, **kwargs):
        """
        :param patience: 训练损失在多少个steps内没有改善后停止训练
        :param threshold: 损失改善的最小阈值，只有当损失降低超过这个值才算"改善"
        """
        self.num_warmup_steps = num_warmup_steps
        self.max_steps = max_steps
        self.patience = patience_stop
        self.threshold = threshold_imporv
        self.best_loss = float("inf")
        self.wait = 0
        self.save_best_model_dir = save_best_model_dir
        # self.optimizer = optim
        # 初始化 Warmup 和 ReduceLROnPlateau
        # self.warmup_scheduler = LambdaLR(optim, lr_lambda=self.warmup_lambda)
        #mode='min', factor=factor, patience=patience, threshold=threshold
        self.plateau_scheduler = ReduceLROnPlateau(optim, **kwargs)

    def warmup_lambda(self, current_step):
        """线性 Warmup 逻辑"""
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        return 1.0  # Warmup 结束后，学习率保持不变，交由 Plateau 控制
    
    # def on_step_end(self, args, state, control,  **kwargs):
    #     pass
        
    def on_log(self, args: transformers.TrainingArguments, state: transformers.TrainerState, 
               control: transformers.TrainerControl, logs=None, lr_scheduler=None, **kwargs):
    

        current_loss = logs['loss']
        if current_loss is not None:

            
            if state.global_step < self.num_warmup_steps:
                # Warmup 阶段
                # self.warmup_scheduler.step()
                print(f"Warmup Step {state.global_step}/{self.num_warmup_steps}, Learning Rate: {kwargs['optimizer'].param_groups[0]['lr']}")
            else:
                # Warmup 结束后，启动 ReduceLROnPlateau
                # self.plateau_scheduler.step(current_loss)
                print(f"Step {state.global_step}/{self.max_steps}, Learning Rate: {kwargs['optimizer'].param_groups[0]['lr']}")
            
            if current_loss < self.best_loss - self.threshold:
                self.best_loss = current_loss
                self.wait = 0              
                control.should_save = True  
            else:
                self.wait += 1
            
            # lr_scheduler.step(current_loss)
            # print(f"Step {state.global_step}, Learning Rate: {lr_scheduler.get_last_lr()}")
            
            if self.wait >= self.patience:
                print(f"Stopping early at step {state.global_step}.")
                control.should_early_stop = True

        return control
