# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Utilities for models."""

import math

import torch

from cuda import load
load()

import fused_quant
class FusedQuant(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, clip_val, clip_valn, num_bits):
        ctx.save_for_backward(input, clip_val, clip_valn)
        ctx.num_bits = num_bits
        return fused_quant.forward_quant(input, clip_val, clip_valn, num_bits)

    @staticmethod
    def backward(ctx, dout):
        input, cv, cvn = ctx.saved_tensors
        num_bits = ctx.num_bits
        di, dcv, dcvn = fused_quant.backward_quant(dout, input, cv, cvn, num_bits)
        return di, dcv, dcvn, None


class QLinear(torch.nn.Linear):
    def __init__(self, init_clip_valp, init_clip_valn, num_bits, *args):
        super().__init__(*args)
        self.clip_val = torch.nn.Parameter(torch.tensor(init_clip_valp))
        self.clip_valn = torch.nn.Parameter(torch.tensor(init_clip_valn))
        self.num_bits = num_bits

    def forward(self, input):
        return torch.nn.functional.linear(FusedQuant.apply(input, self.clip_val, self.clip_valn, self.num_bits), self.weight, self.bias)

