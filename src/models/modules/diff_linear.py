# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2021/12/3
@Desc               :
@Last modified by   : Bao
@Last modified date : 2021/12/3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, num_diffs=1):
        super(DiffLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_normal_(self.base_weight)
        if bias:
            self.base_bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.normal_(self.base_bias)
        else:
            self.base_bias = None

        self.diff_dict = nn.ParameterDict()
        for index in range(num_diffs):
            diff_weight = nn.Parameter(torch.Tensor(out_features, in_features))
            nn.init.xavier_normal_(diff_weight)
            self.diff_dict["weight-{}".format(index)] = diff_weight
            if bias:
                diff_bias = nn.Parameter(torch.Tensor(out_features))
                nn.init.normal_(diff_bias)
                self.diff_dict["bias-{}".format(index)] = diff_bias

    def forward(self, inputs, diff_index):
        weight = self.base_weight + self.diff_dict["weight-{}".format(diff_index)]
        if self.base_bias is not None:
            bias = self.base_bias + self.diff_dict["bias-{}".format(diff_index)]
            outputs = F.linear(inputs, weight, bias)
        else:
            outputs = F.linear(inputs, weight)
        return outputs
