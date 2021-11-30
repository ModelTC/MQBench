```python   
    module = a()
    module = prepare_by_platform(module, BackendType.Vitis)
    print(module(torch.rand(1,2,3,3)))
    enable_calibration(module) 
    print(module(torch.rand(1,2,3,3)))
    enable_quantization(module)
    print(module(torch.rand(1,2,3,3)))
    module.eval() 

    convert_deploy(module, BackendType.Vitis, {'x': [1,2,3,3]}, model_name='validate/conv_no_bias.onnx')
```
Converting models just like others, but this would generate 3 files: 
	1. an onnx file with all fake quantize nodes
	2. an onnx file without any fake quantize nodes
	3. a xmodel file 
