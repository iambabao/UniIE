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


class BertClassifierU(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_tasks = config.num_tasks
        self.task2labels = config.task2labels
        self.task_hidden_size = config.task_hidden_size

        self.bert = BertModel(config)
        self.start_layer = nn.Sequential(
            nn.Linear(config.hidden_size, config.task_hidden_size),
            nn.LayerNorm(config.task_hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.end_layer = nn.Sequential(
            nn.Linear(config.hidden_size, config.task_hidden_size),
            nn.LayerNorm(config.task_hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.Us = nn.ParameterList([
            nn.Parameter(torch.Tensor(len(_) + 1, config.task_hidden_size + 1, config.task_hidden_size + 1))
            for _ in config.task2labels.values()
        ])  # add 1 for bias
        self.loss_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.loss_function = nn.CrossEntropyLoss()

        self.init_weights()

    def forward(
            self,
            task_id,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            length=None,
            labels=None,
    ):
        batch_size = input_ids.shape[0]
        max_seq_length = input_ids.shape[1]
        length_indexes = torch.arange(max_seq_length).expand(batch_size, max_seq_length).to(self.device)

        # mask for real tokens with shape (batch_size, max_seq_length)
        token_mask = torch.less(length_indexes, length.unsqueeze(-1))
        # mask for valid position with shape (batch_size, max_seq_length, max_seq_length)
        position_mask = torch.logical_and(
            token_mask.unsqueeze(-1).expand(-1, -1, max_seq_length),
            token_mask.unsqueeze(-2).expand(-1, max_seq_length, -1),
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,
        )
        sequence_output = outputs[0]  # (batch_size, max_seq_length, hidden_size)
        start_output = self.start_layer(sequence_output)
        start_output = torch.cat([start_output, torch.ones_like(start_output[..., :1])], dim=-1)
        end_output = self.end_layer(sequence_output)
        end_output = torch.cat([end_output, torch.ones_like(end_output[..., :1])], dim=-1)

        task_logits = [
            torch.einsum('xi,oij,yj->oxy', start_output[b_id], self.Us[t_id], end_output[b_id]).permute(1, 2, 0)
            for b_id, t_id in enumerate(task_id)
        ]
        outputs = (task_logits,) + outputs

        if labels is not None:
            labels = labels.reshape(-1, max_seq_length, max_seq_length)

            losses = [
                self.loss_function(
                    self.loss_dropout(task_logits[b_id][position_mask[b_id]]), labels[b_id][position_mask[b_id]]
                ) for b_id in range(len(task_logits))
            ]
            loss = sum(losses) / len(losses)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, ...
