'''
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of BSD 3-Clause License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
BSD 3-Clause License for more details.
'''
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def Binarize(tensor, quant_mode='det', bin=False):
#     if bin:
#         tensor -= 0.5
    if quant_mode=='det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).cuda().add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)

class MaskConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, n_basis, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MaskConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.weight = None
        n_basis = int(n_basis)
        self.n_basis = int(n_basis)
        self.basis_list = nn.Parameter(torch.Tensor(n_basis, in_channels, kernel_size, kernel_size))
        self.mask = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.reset_custome_parameters()
        
    def reset_custome_parameters(self):
        nn.init.kaiming_uniform_(self.basis_list, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mask, a=math.sqrt(5))
        #nn.init.uniform_(self.mask, -1, 1)

    def forward(self, input, bin=False):
        if not hasattr(self.mask, 'org'):
            self.mask.org = self.mask.data.clone()

        self.mask.data = Binarize(self.mask.org, bin=bin)
        conv_weight = torch.mul(self.mask, self.basis_list.repeat(int(self.out_channels/self.n_basis),1,1,1))

        out = F.conv2d(input, conv_weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
    
class MaskConv2dShare(nn.Conv2d):
    def __init__(self, in_channels, out_channels, n_basis, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MaskConv2dShare, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.weight = None
        n_basis = int(n_basis)
        self.n_basis = n_basis
        self.basis_list = nn.Parameter(torch.Tensor(n_basis, in_channels, kernel_size, kernel_size))
        self.mask = nn.Parameter(torch.Tensor(int(out_channels/n_basis), in_channels, kernel_size, kernel_size))
        self.reset_custome_parameters()
        
    def reset_custome_parameters(self):
        nn.init.kaiming_uniform_(self.basis_list, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mask, a=math.sqrt(5))
        #nn.init.uniform_(self.mask, -1, 1)

    def forward(self, input):
        if not hasattr(self.mask,'org'):
            self.mask.org = self.mask.data.clone()
        
        self.mask.data = Binarize(self.mask.org)
        ratio = int(self.out_channels/self.n_basis)
        repeat_mask = torch.cat([self.mask[i].repeat(self.n_basis,1,1,1) for i in range(ratio)], 0)
        conv_weight = torch.mul(repeat_mask, self.basis_list.repeat(ratio,1,1,1))

        out = F.conv2d(input, conv_weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
    
def ortho_loss(model):
    ortho_loss = 0
    for name, param in model.named_parameters():
        if 'mask' in name:
            X = param.view(param.size(0), -1)
            XXT = torch.matmul(X, X.transpose(0,1))/X.size(1)
            I = torch.eye(param.size(0)).cuda()
            ortho_loss += F.mse_loss(I, XXT, size_average=True)
    return ortho_loss

