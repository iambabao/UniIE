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


class BertClassifierW(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_tasks = config.num_tasks
        self.task2labels = config.task2labels
        self.task_hidden_size = config.task_hidden_size

        self.bert = BertModel(config)
        self.start_layer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.end_layer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.start_output_layer = nn.Linear(config.hidden_size, 2)
        self.end_output_layer = nn.Linear(config.hidden_size, 2)
        self.task_layers = nn.ModuleList([
            nn.Linear(2 * config.hidden_size, config.task_hidden_size) for _ in range(config.num_tasks)
        ])
        self.output_layers = nn.ModuleList([
            nn.Linear(config.task_hidden_size, len(_) + 1) for _ in config.task2labels.values()
        ])
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
            start_labels=None,
            end_labels=None,
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
        ).triu()

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,
        )
        sequence_output = outputs[0]  # (batch_size, max_seq_length, hidden_size)
        start_output = self.start_layer(sequence_output)
        end_output = self.end_layer(sequence_output)
        start_logits = self.start_output_layer(start_output)
        end_logits = self.end_output_layer(end_output)
        outputs = (start_logits, end_logits) + outputs

        start_expanded = start_output.unsqueeze(2).expand(-1, -1, max_seq_length, -1)
        end_expanded = end_output.unsqueeze(1).expand(-1, max_seq_length, -1, -1)
        unified_matrix = torch.cat([start_expanded, end_expanded], dim=-1)

        # construct task cube
        # (batch_size, num_tasks, max_seq_length, max_seq_length, task_hidden_size)
        task_cube = torch.stack([task_layer(unified_matrix) for task_layer in self.task_layers], dim=1)

        # construct task matrix
        # (batch_size, max_seq_length, max_seq_length, task_hidden_size)
        index = task_id.view(batch_size, 1, 1, 1, 1).repeat(1, 1, max_seq_length, max_seq_length, self.task_hidden_size)
        task_matrix = torch.gather(task_cube, dim=1, index=index).squeeze(1)

        # batch_size * (max_seq_length, max_seq_length, num_labels)
        task_logits = [self.output_layers[t_id](task_matrix[b_id]) for b_id, t_id in enumerate(task_id)]
        outputs = (task_logits,) + outputs

        if labels is not None:
            labels = labels.reshape(-1, max_seq_length, max_seq_length)

            # mask for golden start and end tokens
            golden_mask = torch.logical_and(
                start_labels.unsqueeze(-1).expand(-1, -1, max_seq_length),
                end_labels.unsqueeze(-2).expand(-1, max_seq_length, -1),
            )
            # mask for predicted start and end tokens
            predicted_mask = torch.logical_and(
                (torch.argmax(start_logits, dim=-1) == 1).unsqueeze(-1).expand(-1, -1, max_seq_length),
                (torch.argmax(end_logits, dim=-1) == 1).unsqueeze(-2).expand(-1, max_seq_length, -1)
            )
            # mask for loss
            final_mask = position_mask & (golden_mask | predicted_mask)

            start_loss = self.loss_function(self.loss_dropout(start_logits)[token_mask], start_labels[token_mask])
            end_loss = self.loss_function(self.loss_dropout(end_logits)[token_mask], end_labels[token_mask])
            losses = []
            for b_id in range(batch_size):
                l = self.loss_function(
                    self.loss_dropout(task_logits[b_id][final_mask[b_id]]), labels[b_id][final_mask[b_id]]
                )
                if not torch.isnan(l):
                    losses.append(l)
            if len(losses) == 0:
                losses.append(0.0)
            loss = sum(losses) / len(losses) + start_loss + end_loss
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, ...
