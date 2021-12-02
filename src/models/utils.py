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
