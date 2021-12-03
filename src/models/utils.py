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


def merge_by_attention(task_cube, u, length, task_id):
    """

    :param task_cube: (batch_size, num_tasks, max_num_tokens, max_num_tokens, task_hidden_size)
    :param u: (task_hidden_size, task_hidden_size)
    :param length: (batch_size,)
    :param task_id: (batch_size,)
    :return:
    """

    outputs = []
    for cube, max_length, t_id in zip(task_cube, length, task_id):
        cube = cube[:, :max_length, :max_length, :]  # (num_tasks, max_length, max_length, task_hidden_size)
        # TODO: mask unused cell
        task_features = torch.mean(cube, dim=(1, 2))  # (num_tasks, task_hidden_size)
        scores = torch.einsum("ik,kk,jk->ij", task_features, u, task_features)  # (num_tasks, num_tasks)
        attention = torch.softmax(scores[t_id], dim=-1)  # (num_tasks,)
        outputs.append(torch.sum(attention.view(-1, 1, 1, 1) * cube, dim=0))
    return torch.stack(outputs, dim=0)


def linearized_merge_by_attention(task_cube, u, matrix_mask, length, task_id):
    """

    :param task_cube: (batch_size, num_tasks, X, hidden_size)
    :param u: (hidden_size, hidden_size)
    :param matrix_mask: (batch_size, X)
    :param length: (batch_size,)
    :param task_id: (batch_size)
    :return:
    """

    outputs = []
    for cube, mask, max_length, t_id in zip(task_cube, matrix_mask, length, task_id):
        # (i, k, m) (m, n) (j, k, n) --> (i, j, k)
        scores = torch.einsum("ikm,mn,jkn->ijk", cube, u, cube)  # (num_tasks, num_tasks, X)
        attention = torch.softmax(scores[t_id], dim=0)  # (num_tasks, X)
        outputs.append(torch.sum(attention.unsqueeze(-1) * cube, dim=0))
    return torch.stack(outputs, dim=0)


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
