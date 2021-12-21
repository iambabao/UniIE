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


class VariantA(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_tasks = config.num_tasks
        self.task2labels = config.task2labels
        self.task_hidden_size = config.task_hidden_size

        self.bert = BertModel(config)
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

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # (batch_size, max_seq_length, hidden_size)
        sequence_output = outputs["last_hidden_state"]
        # (batch_size, max_num_tokens, hidden_size)
        token_embeddings = build_token_embedding(sequence_output, token_mapping)

        start_embeddings = self.unified_start_layer(token_embeddings)
        start_embeddings = [layer(start_embeddings) for layer in self.task_start_layers]
        start_embeddings = torch.stack(start_embeddings, dim=1)
        start_embeddings = attentive_select(start_embeddings, self.start_u, task_id)

        end_embeddings = self.unified_end_layer(token_embeddings)
        end_embeddings = [layer(end_embeddings) for layer in self.task_end_layers]
        end_embeddings = torch.stack(end_embeddings, dim=1)
        end_embeddings = attentive_select(end_embeddings, self.end_u, task_id)

        start_logits = self.output_start_layer(start_embeddings)
        end_logits = self.output_end_layer(end_embeddings)
        task_logits = [
            torch.einsum("im,mnk,jn->ijk", start_embeddings[b_id], self.Us[t_id], end_embeddings[b_id])
            for b_id, t_id in enumerate(task_id)
        ]

        outputs["position_mask"] = position_mask
        outputs["token_embeddings"] = token_embeddings
        outputs["start_embeddings"] = start_embeddings
        outputs["end_embeddings"] = end_embeddings
        outputs["start_logits"] = start_logits
        outputs["end_logits"] = end_logits
        outputs["task_logits"] = task_logits

        if labels is not None:
            labels = labels.reshape(-1, max_num_tokens, max_num_tokens)

            start_loss = self.loss_function(self.loss_dropout(start_logits)[token_mask], start_labels[token_mask])
            end_loss = self.loss_function(self.loss_dropout(end_logits)[token_mask], end_labels[token_mask])
            losses = [
                self.loss_function(
                    self.loss_dropout(task_logits[b_id][position_mask[b_id]]), labels[b_id][position_mask[b_id]]
                ) for b_id in range(batch_size)
            ]
            loss = sum(losses) / len(losses) + 0.5 * (start_loss + end_loss)
            outputs["start_loss"] = start_loss
            outputs["end_loss"] = end_loss
            outputs["loss"] = loss

        return outputs  # (loss), logits, ...
