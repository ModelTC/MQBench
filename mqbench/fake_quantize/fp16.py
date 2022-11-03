import torch


class Fp16FakeQuantize():
    def __init__(self):
        pass

    @torch.jit.export
    def extra_repr(self):
        return 'Fp16FakeQuantize'

    def forward(self, X):
        #调用自定义torch c++ op将fp32的X转为fp16后再转会fp32，引入误差

class BF16FakeQuantize():
    def __init__(self):
        pass

    @torch.jit.export
    def extra_repr(self):
        return 'BF16FakeQuantize'

    def forward(self, X):
        #fp32的X转为bf16后再转会fp32，引入误差