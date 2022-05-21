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

from src.models.modules import DiffLinear
from src.models.utils import build_token_embedding


class VariantB(BertPreTrainedModel):
    """
    Model for multi-task learning with:
        - diff-parameter
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
        self.diff_start_layer = DiffLinear(config.hidden_size, config.task_hidden_size, num_diffs=config.num_tasks)
        self.diff_end_layer = DiffLinear(config.hidden_size, config.task_hidden_size, num_diffs=config.num_tasks)
        self.output_start_layer = nn.Linear(config.task_hidden_size, 2)
        self.output_end_layer = nn.Linear(config.task_hidden_size, 2)
        self.Us = nn.ParameterList([
            nn.Parameter(torch.Tensor(config.task_hidden_size, config.task_hidden_size, len(_) + 1))
            for _ in config.task2labels.values()
        ])
        self.loss_dropout = nn.Dropout(config.hidden_dropout_prob * 2)
        self.loss_function = nn.CrossEntropyLoss()

        # initialize manually added parameters
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
        prior = batch_inputs.get("prior")

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
        start_embedding = [self.diff_start_layer(start_embedding[b_id], t_id) for b_id, t_id in enumerate(task_id)]
        start_embedding = torch.stack(start_embedding, dim=0)

        end_embedding = self.unified_end_layer(token_embedding)
        end_embedding = [self.diff_end_layer(end_embedding[b_id], t_id) for b_id, t_id in enumerate(task_id)]
        end_embedding = torch.stack(end_embedding, dim=0)

        start_logits = self.output_start_layer(start_embedding)
        end_logits = self.output_end_layer(end_embedding)
        task_logits = [
            torch.einsum("im,mnk,jn->ijk", start_embedding[b_id], self.Us[t_id], end_embedding[b_id])
            for b_id, t_id in enumerate(task_id)
        ]

        outputs["position_mask"] = position_mask
        outputs["token_embedding"] = token_embedding
        outputs["start_embedding"] = start_embedding
        outputs["end_embedding"] = end_embedding
        outputs["start_logits"] = start_logits
        outputs["end_logits"] = end_logits
        outputs["task_logits"] = task_logits

        if labels is not None:
            labels = labels.reshape(-1, max_num_tokens, max_num_tokens)

            start_loss = self.loss_function(self.loss_dropout(start_logits)[token_mask], start_labels[token_mask])
            end_loss = self.loss_function(self.loss_dropout(end_logits)[token_mask], end_labels[token_mask])
            position_loss = 0.5 * (start_loss + end_loss)

            task_loss = [
                self.loss_function(
                    self.loss_dropout(task_logits[b_id][position_mask[b_id]]), labels[b_id][position_mask[b_id]]
                ) for b_id in range(batch_size)
            ]
            task_loss = sum(task_loss) / len(task_loss)

            loss = position_loss + task_loss
            if prior is not None:
                kd_function = nn.KLDivLoss(reduction="sum")
                kd_loss = [
                    kd_function(torch.log_softmax(task_logits[b_id][position_mask[b_id]], dim=-1), prior[b_id])
                    for b_id in range(batch_size)
                ]
                kd_loss = sum(kd_loss) / len(kd_loss)
                loss = loss + kd_loss
                outputs["kd_loss"] = kd_loss

            outputs["position_loss"] = position_loss
            outputs["task_loss"] = task_loss
            outputs["loss"] = loss

        return outputs
