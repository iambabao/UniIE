# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/10/9
@Desc               :
@Last modified by   : Bao
@Last modified date : 2021/4/25
"""

import torch
import torch.nn.functional as F


def build_token_embedding(inputs, mapping):
    """

    :param inputs: (batch_size, max_seq_length, hidden_size)
    :param mapping: (batch_size, max_num_tokens, 2)
    :return:
    """

    reshape = list(mapping.shape[:-1]) + [inputs.shape[-1]]
    start_index, end_index = mapping[..., 0], mapping[..., 1]

    start_index = start_index.unsqueeze(-1).expand(*reshape)
    start_feature = torch.gather(inputs, dim=1, index=start_index)

    end_index = end_index.unsqueeze(-1).expand(*reshape)
    end_feature = torch.gather(inputs, dim=1, index=end_index)

    return (start_feature + end_feature) / 2.0


def linearize_matrix(matrix, mask):
    """

    :param matrix: (batch_size, max_seq_length, max_seq_length, hidden_size)
    :param mask: (batch_size, max_seq_length, max_seq_length)
    :return:
    """

    linearized_matrix, linearized_mask = [], []
    for index in range(matrix.shape[1]):
        linearized_matrix.append(matrix[:, index, :-index, :])
        linearized_mask.append(mask[:, index, :-index])
    linearized_matrix = torch.cat(linearized_matrix, dim=1)
    linearized_mask = torch.cat(linearized_mask, dim=1)
    return linearized_matrix, linearized_mask


def merge_by_select(task_cube, task_id):
    """

    :param task_cube: (batch_size, num_tasks, max_num_tokens, max_num_tokens, task_hidden_size)
    :param task_id: (batch_size,)
    :return:
    """

    outputs = []
    for cube, t_id in zip(task_cube, task_id):
        outputs.append(cube[t_id])
    return torch.stack(outputs, dim=0)


def index_select(features, task_id):
    """

    :param features: (batch_size, num_tasks, max_num_tokens, hidden_size)
    :param task_id: (batch_size,)
    :return:
    """

    outputs = []
    for feature, t_id in zip(features, task_id):
        outputs.append(feature[t_id])
    return torch.stack(outputs, dim=0)


def attentive_select(features, u, task_id):
    """

    :param features: (batch_size, num_tasks, max_num_tokens, hidden_size)
    :param u: (hidden_size, hidden_size)
    :param task_id: (batch_size,)
    :return:
    """

    outputs = []
    for feature, t_id in zip(features, task_id):
        scores = torch.einsum("ikm,mn,jkn->ijk", feature, u, feature)  # (num_tasks, num_tasks, max_num_tokens)
        attention = torch.softmax(scores[t_id], dim=0)  # (num_tasks, max_num_tokens)
        outputs.append(torch.sum(attention.unsqueeze(-1) * feature, dim=0))
    return torch.stack(outputs, dim=0)


def consistency_loss_function(logits, target):
    x = torch.log_softmax(logits, dim=-1)
    y = torch.softmax(target, dim=-1)
    return -torch.mean(torch.sum((x * y), dim=-1))


def ce_loss(logits, labels, mask=None):
    """

    Args:
        logits: (batch, ..., num_labels)
        labels: (batch, ...)
        mask: (batch, ...)

    Returns:

    """

    num_labels = logits.shape[-1]
    loss = F.cross_entropy(logits.view(-1, num_labels), labels.view(-1).type(torch.long), reduction='none')
    if mask is not None:
        if torch.sum(mask) != 0:
            loss = torch.sum(mask.view(-1) * loss) / torch.sum(mask)
        else:
            loss = 0.0
    else:
        loss = torch.mean(loss)
    return loss
