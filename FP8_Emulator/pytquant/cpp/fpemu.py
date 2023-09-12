#------------------------------------------------------------------------------ 
# Copyright (c) 2023, Intel Corporation - All rights reserved. 
# This file is part of FP8-Emulation-Toolkit
#
# SPDX-License-Identifier: BSD-3-Clause
#------------------------------------------------------------------------------
# Naveen Mellempudi (Intel Corporation)
#------------------------------------------------------------------------------

import math
from torch import nn
from torch.autograd import Function
import torch
import numpy
import fpemu_cpp

from enum import Enum

torch.manual_seed(42)

"""
    NONE
    E5M2_RTZ
    E5M2_STOCHASTIC
    E5M2_RNE
    E5M2_RNAZ
    E5M2_RNTZ
    E5M2_RPINF
    E5M2_RNINF
    E5M2_DAZ_STOCHASTIC
    E5M2_DAZ_RNE
    E5M2_DAZ_RNAZ
    E5M2_DAZ_RNTZ
    BFLOAT16_STOCHASTIC
    BFLOAT16_RNE
    FLOAT16_RNE
    FLOAT16_STOCHASTIC
    FLOAT16_DAZ_RNE
    E4M3_RNE
    E4M3_STOCHASTIC
"""

class FPEmuOp(Function):
    @staticmethod
    def forward(ctx, input, mode='NONE', inplace=False, scale=1.0, blocknorm=False, blocksize=1):
        if mode == 'NONE' :
            ctx.mark_dirty(input)
            return input
        else :
            if input.is_sparse :
                input = input.coalesce()
                size = input.values().nelement()
                if inplace == True:
                    outputs = fpemu_cpp.forward(input._values().contiguous(), mode, size, inplace, scale, blocknorm, blocksize)
                    output = input
                else :
                    outputs = fpemu_cpp.forward(input._values().contiguous(), mode, size, inplace, scale, blocknorm, blocksize) 
                    output = torch.sparse.FloatTensor(input.indices(), outputs[0], input.size())
            else :
                input = input.cpu()
                size = input.nelement()
                outputs = fpemu_cpp.forward(input.contiguous(), mode, size, inplace, scale, blocknorm, blocksize)
                output = outputs[0]

            if inplace == True:
                ctx.mark_dirty(input)

            return output

    @staticmethod
    def backward(ctx, output_grad):
        # straight-through estimator
        return output_grad, None, None, None, None
