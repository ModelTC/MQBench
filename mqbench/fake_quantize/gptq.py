import torch
from torch.nn.parameter import Parameter
from torch.quantization.observer import MovingAverageMinMaxObserver

from mqbench.fake_quantize.quantize_base import QuantizeBase, _version_under_1100 
from mqbench.utils.hook import PerChannelLoadHook
from mqbench.fake_quantize import global_var

import transformers
import math
import time

DEBUG = False 

def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

class GPTQFakeQuantize(QuantizeBase):
    '''
    This is gptq method mqbench version.
    '''

    def __init__(self, observer, **observer_kwargs):
        super(GPTQFakeQuantize, self).__init__(observer, **observer_kwargs)
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        self.is_gptq_valid = False
        self.layer_name = None
        self.rows = None
        self.columns = None
        # self.H = None
        self.nsamples = 0
        self.layer_module = None
        self.dev = torch.device('cpu')
        self.register_buffer('H', torch.tensor([0], dtype=torch.float))


    def forward(self, X):
        # print('This FakeQuantizer is "', self.layer_name, '" and input is "', self.layer_input, '" and type is "', self.layer_type, '" is weight? ', self.is_weight)
        self.dev = X.device
        x_ori = X
        # print("GPTQFakeQuantize - forward")
        if self.observer_enabled[0] == 1:
            if (self.is_gptq_valid):
                W = X.data.clone()
                if isinstance(self.layer_module, torch.nn.Conv2d):
                    W = W.flatten(1)
                if isinstance(self.layer_module, transformers.Conv1D):
                    W = W.t()
                self.rows = W.shape[0]
                self.columns = W.shape[1]
                self.H = torch.zeros((W.shape[1], W.shape[1]), device=self.dev)

            self.activation_post_process(X.detach())
            # All is per layer
            _scale, _zero_point = self.activation_post_process.calculate_qparams()
            _scale = _scale.to(self.scale.device)
            _zero_point = _zero_point.to(self.zero_point.device)

            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)

            self.scale.data.copy_(_scale)
            self.zero_point.data.copy_(_zero_point.float())

        if self.fake_quant_enabled[0] == 1:
            # Use GPTQ
            if (self.is_gptq_valid):
                with torch.no_grad():
                    self.input = global_var.get_value(self.layer_name+'.inp')
                    self.output = global_var.get_value(self.layer_name+'.out')
                    if (self.input.device != self.dev or self.output.device != self.dev):
                        self.input = self.input.to(self.dev)
                        self.output = self.output.to(self.dev)
                    self.weight = X
                    self.add_batch()
                    X = self.fasterquant()
            # Use FixedFakeQuantize per Tensor
            else:
                X = torch.fake_quantize_per_tensor_affine(
                    X, self.scale.item(), int(self.zero_point.item()),
                    self.quant_min, self.quant_max)
        return X
    
    @torch.jit.export
    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={}, ' \
               'quant_min={}, quant_max={}, dtype={}, qscheme={}, ch_axis={}, ' \
               'scale={}, zero_point={}'.format(
                   self.fake_quant_enabled, self.observer_enabled,
                   self.quant_min, self.quant_max,
                   self.dtype, self.qscheme, self.ch_axis, self.scale if self.ch_axis == -1 else 'List[%s]' % str(self.scale.shape),
                   self.zero_point if self.ch_axis == -1 else 'List')

    def add_batch(self):
        inp = self.input
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]

        if isinstance(self.layer_module, torch.nn.Linear) or isinstance(self.layer_module, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer_module, torch.nn.Conv2d):
            unfold = torch.nn.Unfold(
                self.layer_module.kernel_size,
                dilation=self.layer_module.dilation,
                padding=self.layer_module.padding,
                stride=self.layer_module.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

        self.H *= self.nsamples / (self.nsamples + tmp) # always zero ?
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        inp_inpt = inp.matmul(inp.t())
        self.H += inp_inpt
        # inp_inpt.to(torch.device('cpu'))
        # torch.cuda.empty_cache()
        del inp_inpt
        # print(self.H.size())

    def fasterquant(self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False):
        # W = self.layer.weight.data.clone()
        # if isinstance(self.layer,torch.nn.Conv2d):
        W = self.weight.data.clone()

        if isinstance(self.layer_module, torch.nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer_module, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        # Calculate by Observer
        # if not self.quantizer.ready():
        #     self.quantizer.find_params(W, weight=True)

        H = self.H
        # del self.H

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # if actorder:
        #     perm = torch.argsort(torch.diag(H), descending=True)
        #     W = W[:, perm]
        #     H = H[perm][:, perm]

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize): # 以步长 blocksize (128) 循环到 columns
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                # Find by observer
                # if groupsize != -1:
                #     if (i1 + i) % groupsize == 0:
                #         self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)

                q = quantize(
                    w, self.scale, self.zero_point, self.quant_max
                ).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        # print('time %.2f' % (time.time() - tick))
        # print('error', torch.sum(Losses).item())

        if isinstance(self.layer_module, transformers.Conv1D):
            Q = Q.t()
        # self.weight.data = Q.reshape(self.weight.shape).to(self.weight.data.dtype)
        W = Q.reshape(self.weight.shape).to(self.weight.data.dtype) # fixed fakequantize 并没有重置 parameter weight

        # if DEBUG:
        #     print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        self.layer_module.weight.data = W
        # if ('Conv2d' in self.layer_type):
            # print(torch.sum((self.layer_module(self.input) - self.layer_out) ** 2))
        # if ('Linear' in self.layer_type):
        # print(torch.sum((self.layer_module(self.input) - self.output) ** 2))
        # del self.layer_module
        
        return W
