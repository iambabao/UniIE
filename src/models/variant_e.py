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

from src.models.utils import build_token_embedding, attentive_select


class VariantE(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_tasks = config.num_tasks
        self.task2labels = config.task2labels
        self.task_hidden_size = config.task_hidden_size

        self.bert = BertModel(config)
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
        self.Us = nn.ParameterList([
            nn.Parameter(torch.Tensor(config.task_hidden_size, config.task_hidden_size, len(_) + 1))
            for _ in config.task2labels.values()
        ])
        self.loss_dropout = nn.Dropout(config.hidden_dropout_prob * 2)
        self.loss_function = nn.CrossEntropyLoss()

        # initialize manually added parameters
        nn.init.xavier_normal_(self.start_u)
        nn.init.xavier_normal_(self.end_u)
        for _ in self.Us:
            nn.init.xavier_normal_(_)

        # initialize with HuggingFace API
        self.init_weights()

    def forward(
            self,
            task_id,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            token_mapping=None,
            length=None,
            start_labels=None,
            end_labels=None,
            labels=None,
    ):
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

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,
        )
        # (batch_size, max_seq_length, hidden_size)
        sequence_output = outputs[0]
        # (batch_size, max_num_tokens, hidden_size)
        token_embeddings = build_token_embedding(sequence_output, token_mapping)

        start_embeddings = torch.stack([layer(token_embeddings) for layer in self.task_start_layers], dim=1)
        start_embeddings = attentive_select(start_embeddings, self.start_u, task_id)

        end_embeddings = torch.stack([layer(token_embeddings) for layer in self.task_end_layers], dim=1)
        end_embeddings = attentive_select(end_embeddings, self.end_u, task_id)

        start_logits = self.output_start_layer(start_embeddings)
        end_logits = self.output_end_layer(end_embeddings)
        task_logits = [
            torch.einsum("im,mnk,jn->ijk", start_embeddings[b_id], self.Us[t_id], end_embeddings[b_id])
            for b_id, t_id in enumerate(task_id)
        ]
        outputs = (task_logits,) + outputs

        if labels is not None:
            labels = labels.reshape(-1, max_num_tokens, max_num_tokens)

            start_loss = self.loss_function(self.loss_dropout(start_logits)[token_mask], start_labels[token_mask])
            end_loss = self.loss_function(self.loss_dropout(end_logits)[token_mask], end_labels[token_mask])
            losses = [
                self.loss_function(
                    self.loss_dropout(task_logits[b_id][position_mask[b_id]]), labels[b_id][position_mask[b_id]]
                ) for b_id in range(batch_size)
            ]
            loss = sum(losses) / len(losses) + start_loss + end_loss
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, ...
