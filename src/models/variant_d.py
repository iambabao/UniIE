# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/7/26
@Desc               : 
@Last modified by   : Bao
@Last modified date : 2021/5/8
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

from src.models.utils import build_token_embedding, attentive_select, consistency_loss_function


class VariantD(BertPreTrainedModel):
    """
    Model for multi-task learning with:
        - attention mechanism
        - consistency constrain
    """

    def __init__(self, config):
        super().__init__(config)

        self.num_tasks = config.num_tasks
        self.task2labels = config.task2labels
        self.task_hidden_size = config.task_hidden_size

        self.bert = BertModel(config, add_pooling_layer=False)
        self.unified_start_layer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob * 4),
        )
        self.unified_end_layer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob * 4),
        )
        self.task_start_layers = nn.ModuleList([
            nn.Linear(config.hidden_size, config.task_hidden_size) for _ in range(config.num_tasks)
        ])
        self.task_end_layers = nn.ModuleList([
            nn.Linear(config.hidden_size, config.task_hidden_size) for _ in range(config.num_tasks)
        ])
        self.start_u = nn.Parameter(torch.Tensor(config.task_hidden_size, config.task_hidden_size))
        self.end_u = nn.Parameter(torch.Tensor(config.task_hidden_size, config.task_hidden_size))
        self.output_start_layer = nn.Linear(config.task_hidden_size, 2)
        self.output_end_layer = nn.Linear(config.task_hidden_size, 2)
        self.Us_raw = nn.ParameterList([
            nn.Parameter(torch.Tensor(config.task_hidden_size, config.task_hidden_size, len(_) + 1))
            for _ in config.task2labels.values()
        ])
        self.Us_att = nn.ParameterList([
            nn.Parameter(torch.Tensor(config.task_hidden_size, config.task_hidden_size, len(_) + 1))
            for _ in config.task2labels.values()
        ])
        self.loss_dropout = nn.Dropout(config.hidden_dropout_prob * 2)
        self.loss_function = nn.CrossEntropyLoss()
        self.kl_function = consistency_loss_function

        # initialize manually added parameters
        nn.init.xavier_normal_(self.start_u)
        nn.init.xavier_normal_(self.end_u)
        for _ in self.Us_raw:
            nn.init.xavier_normal_(_)
        for _ in self.Us_att:
            nn.init.xavier_normal_(_)

        # initialize with HuggingFace API
        self.init_weights()

    def forward(self, batch_inputs):
        task_id = batch_inputs.get("task_id")
        input_ids = batch_inputs.get("input_ids")
        attention_mask = batch_inputs.get("attention_mask")
        token_type_ids = batch_inputs.get("token_type_ids")
        token_mapping = batch_inputs.get("token_mapping")
        length = batch_inputs.get("length")
        start_labels = batch_inputs.get("start_labels")
        end_labels = batch_inputs.get("end_labels")
        labels = batch_inputs.get("labels")

        batch_size = token_mapping.shape[0]
        max_num_tokens = token_mapping.shape[1]
        length_indexes = torch.arange(max_num_tokens).expand(batch_size, max_num_tokens).to(self.device)

        # mask for real tokens with shape (batch_size, max_num_tokens)
        token_mask = torch.less(length_indexes, length.unsqueeze(-1))
        # mask for valid positions with shape (batch_size, max_num_tokens, max_num_tokens)
        position_mask = torch.logical_and(
            token_mask.unsqueeze(-1).expand(-1, -1, max_num_tokens),
            token_mask.unsqueeze(-2).expand(-1, max_num_tokens, -1),
        ).triu()

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # (batch_size, max_seq_length, hidden_size)
        sequence_output = outputs["last_hidden_state"]
        # (batch_size, max_num_tokens, hidden_size)
        token_embedding = build_token_embedding(sequence_output, token_mapping)

        start_embedding = self.unified_start_layer(token_embedding)
        start_embedding = [layer(start_embedding) for layer in self.task_start_layers]
        start_embedding = torch.stack(start_embedding, dim=1)
        start_embedding_raw = torch.stack([start_embedding[b_id][t_id] for b_id, t_id in enumerate(task_id)], dim=0)
        start_embedding_att = attentive_select(start_embedding, self.start_u, task_id)

        end_embedding = self.unified_end_layer(token_embedding)
        end_embedding = [layer(end_embedding) for layer in self.task_end_layers]
        end_embedding = torch.stack(end_embedding, dim=1)
        end_embedding_raw = torch.stack([end_embedding[b_id][t_id] for b_id, t_id in enumerate(task_id)], dim=0)
        end_embedding_att = attentive_select(end_embedding, self.end_u, task_id)

        start_logits = self.output_start_layer(start_embedding_att)
        end_logits = self.output_end_layer(end_embedding_att)
        task_logits_raw = [
            torch.einsum("im,mnk,jn->ijk", start_embedding_raw[b_id], self.Us_raw[t_id], end_embedding_raw[b_id])
            for b_id, t_id in enumerate(task_id)
        ]
        task_logits_att = [
            torch.einsum("im,mnk,jn->ijk", start_embedding_att[b_id], self.Us_att[t_id], end_embedding_att[b_id])
            for b_id, t_id in enumerate(task_id)
        ]

        outputs["position_mask"] = position_mask
        outputs["token_embedding"] = token_embedding
        outputs["start_embedding"] = start_embedding
        outputs["end_embedding"] = end_embedding
        outputs["start_logits"] = start_logits
        outputs["end_logits"] = end_logits
        outputs["task_logits"] = task_logits_att

        if labels is not None:
            labels = labels.reshape(-1, max_num_tokens, max_num_tokens)

            start_loss = self.loss_function(self.loss_dropout(start_logits)[token_mask], start_labels[token_mask])
            end_loss = self.loss_function(self.loss_dropout(end_logits)[token_mask], end_labels[token_mask])
            position_loss = start_loss + end_loss

            task_loss_raw = [
                self.loss_function(
                    self.loss_dropout(task_logits_raw[b_id][position_mask[b_id]]), labels[b_id][position_mask[b_id]]
                ) for b_id in range(batch_size)
            ]
            task_loss_raw = sum(task_loss_raw) / len(task_loss_raw)
            task_loss_att = [
                self.loss_function(
                    self.loss_dropout(task_logits_att[b_id][position_mask[b_id]]), labels[b_id][position_mask[b_id]]
                ) for b_id in range(batch_size)
            ]
            task_loss_att = sum(task_loss_att) / len(task_loss_att)
            task_loss = task_loss_raw + task_loss_att

            consistency_loss = [
                self.kl_function(
                    task_logits_raw[b_id][position_mask[b_id]],
                    task_logits_att[b_id][position_mask[b_id]].detach(),
                ) for b_id in range(batch_size)
            ]
            consistency_loss = 0.5 * sum(consistency_loss) / len(consistency_loss)

            loss = consistency_loss + task_loss + position_loss
            outputs["position_loss"] = position_loss
            outputs["task_loss"] = task_loss
            outputs["consistency_loss"] = consistency_loss
            outputs["loss"] = loss

        return outputs
