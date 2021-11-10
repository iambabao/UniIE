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


def ce_loss_with_split(logits, labels, mask=None):
    """

    Args:
        logits: batch * (..., num_labels)
        labels: (batch, ...)
        mask: (batch, ...)

    Returns:

    """

    losses = []
    for index in range(len(logits)):
        num_labels = logits[index].shape[-1]
        loss = F.cross_entropy(logits[index].view(-1, num_labels), labels[index].view(-1).type(torch.long), reduction='none')
        if mask is not None:
            if torch.sum(mask[index]) != 0:
                loss = torch.sum(mask[index].view(-1) * loss) / torch.sum(mask[index])
            else:
                loss = 0.0
        else:
            loss = torch.mean(loss)
        losses.append(loss)
    return sum(losses) / len(losses)
