# 模型测试汇总

config 1:

```python
weight:     (FixedFakeQuant, MinMaxObserver,    sys=True,  per_channel=True)
activation: (FixedFakeQuant, EMAMinMaxObserver, sys=False, per_channel=False)
```

PTQ 结果：

| Model           | Acc@fp32 | Acc@8bit        | SG-Acc@fp32 | SG-Acc@uint8         |
| --------------- | -------- | --------------- | ----------- | -------------------- |
| EfficientnetB0  | 76.104   | 71.052          | 76.048      |  75.61(67.292)       |


> 75.61 是全部 conv 使用 per-channel, 67.292 只是 DepthWiseConv 使用 per-channel


QAT 之后：


| Model           | Acc@fp32 | Acc@8bit | SG-Acc@fp32 | SG-Acc@uint8 | SG-Acc@QAT-uint8 |
| --------------- | -------- | -------- | ----------- | ------------ | ---------------- |
| EfficientnetB0  | 76.104   | 76.32    | 75.75       | 75.69        |  75.47           |


config 2:

```python
weight:     (LearnableFakeQuantize, MinMaxObserver,    sys=False,  per_channel=False)
activation: (LearnableFakeQuantize, EMAMinMaxObserver, sys=False,  per_channel=False)
```

| Model           | Acc@fp32 | Acc@8bit        | SG-Acc@fp32 | SG-Acc@uint8 |
| --------------- | -------- | --------------- | ----------- | ------------ |
| EfficientnetB0  | 76.104   |                 | 76.048      |  8.940       |
| squeezenet1_0   | 58.088   | 52.164/56.980   | 58.080      |  56.946      |


QAT 之后：

| Model           | Acc@fp32 | Acc@8bit        | SG-Acc@fp32 | SG-Acc@uint8 | SG-Acc@QAT-uint8 |
| --------------- | -------- | --------------- | ----------- | ------------ | ---------------- |
| EfficientnetB0  | 76.104   | 73.51           | -           |  -           | 73.552           |
| squeezenet1_0   | 58.088   |                 |             |              |                  |


## MQBench 量化具体流程

### 1. 合并多个模块

为了量化操作的简便性，可以把 `Conv+BN+Relu` 这样的多级 Module 模块合并成一个单独的 Module，可以这样操作的原因是在推理端的量化就是把 `Conv+BN+Relu` 融合在一起，只在 Conv 前进行量化，以及在 Relu 后进行反量化。

当你有自己的模块需要合并时，扩展是比较麻烦的

### 2. Float Module 替换成 Qat Module

这步操作就是对类似 Conv 模块的 weights 参数进行量化反量化操作，这一步比较简单，只需一一对应进行替换操作，torch 默认的模块映射如下：

```python
# Default map for swapping float module to qat modules
DEFAULT_QAT_MODULE_MAPPINGS : Dict[Callable, Any] = {
    nn.Conv2d: nnqat.Conv2d,
    nn.Linear: nnqat.Linear,
    nn.modules.linear._LinearWithBias: nnqat.Linear,
    # Intrinsic modules:
    nni.ConvBn1d: nniqat.ConvBn1d,
    nni.ConvBn2d: nniqat.ConvBn2d,
    nni.ConvBnReLU1d: nniqat.ConvBnReLU1d,
    nni.ConvBnReLU2d: nniqat.ConvBnReLU2d,
    nni.ConvReLU2d: nniqat.ConvReLU2d,
    nni.LinearReLU: nniqat.LinearReLU
}
```

可以看到 `nn.Conv2d: nnqat.Conv2d` 就是把 `nn.Conv2d` 替换成 `nnqat.Conv2d` 模块，而 `nni.ConvBnReLU2d: nniqat.ConvBnReLU2d` 就是先执行了第一个步骤，把 `Conv+BN+Relu` 这样的多级 Module 模块合并成一个单独的 `nni.ConvBnReLU2d` 模块，然后进行替换成 `nniqat.ConvBnReLU2d` 模块。

这是 MQBench 扩展的映射列表：

```python
QAT_MODULE_MAPPINGS = {
    torch.nn.modules.conv.Conv2d: torch.nn.qat.modules.conv.Conv2d, 
    torch.nn.modules.linear.Linear: torch.nn.qat.modules.linear.Linear, 
    torch.nn.modules.linear._LinearWithBias: torch.nn.qat.modules.linear.Linear, 
    torch.nn.intrinsic.modules.fused.ConvBn1d: torch.nn.intrinsic.qat.modules.conv_fused.ConvBn1d, 
    torch.nn.intrinsic.modules.fused.ConvBn2d: torch.nn.intrinsic.qat.modules.conv_fused.ConvBn2d, 
    torch.nn.intrinsic.modules.fused.ConvBnReLU1d: torch.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU1d, 
    torch.nn.intrinsic.modules.fused.ConvBnReLU2d: torch.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU2d, 
    torch.nn.intrinsic.modules.fused.ConvReLU2d: torch.nn.intrinsic.qat.modules.conv_fused.ConvReLU2d, 
    torch.nn.intrinsic.modules.fused.LinearReLU: torch.nn.intrinsic.qat.modules.linear_relu.LinearReLU, 
    torch.nn.modules.conv.ConvTranspose2d: mqbench.nn.qat.modules.deconv.ConvTranspose2d, 
    mqbench.nn.intrinsic.modules.fused.LinearBn1d: mqbench.nn.intrinsic.qat.modules.linear_fused.LinearBn1d, 
    mqbench.nn.intrinsic.modules.fused.ConvTransposeBn2d: mqbench.nn.intrinsic.qat.modules.deconv_fused.ConvTransposeBn2d, 
    mqbench.nn.intrinsic.modules.fused.ConvTransposeReLU2d: mqbench.nn.intrinsic.qat.modules.deconv_fused.ConvTransposeReLU2d, 
    mqbench.nn.intrinsic.modules.fused.ConvTransposeBnReLU2d: mqbench.nn.intrinsic.qat.modules.deconv_fused.ConvTransposeBnReLU2d,
    mqbench.nn.intrinsic.modules.fused.ConvFreezebn2d: mqbench.nn.intrinsic.qat.modules.freezebn.ConvFreezebn2d, 
    mqbench.nn.intrinsic.modules.fused.ConvFreezebnReLU2d: mqbench.nn.intrinsic.qat.modules.freezebn.ConvFreezebnReLU2d
}
```

### 3. 