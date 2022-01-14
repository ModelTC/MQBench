Develop PTQ with MQBench
========================

Assume we get a PTQ algorithm, which needs certain layers' input and certain layers' output for calibration. Also, some calib data provided. And thanks to the prepared model, we do not have to be disturbed by quant/calib/float mode choose. There are only a few steps to imply a PTQ in MQBench. 

1. A fake quantizer. 
2. Data hooks to get intra-network feature maps.
3. A loss function used in calibration. 


Like stated in :ref:`add_a_backend_to_mqbench`, a self-defined quantizer may be required for the PTQ. 

Usually PTQ will adjust the weight via some quantization affine and backward in calibration, whichs need intra-network feature maps. We provided a hook ``mqbench.utils.hooks.DataSaverHook`` to catch input/output of a certain module. Just call it with ``torch.nn.Module.register_forward_hook`` like this, and similarly you can catch the gradients input/outputs: 

.. code-block:: python 
    :linenos:

    def save_inp_oup_data(model: GraphModule, module: Module, cali_data: list, store_inp=True, store_oup=True):
        assert (not store_inp or not store_oup)
        data_saver = DataSaverHook(store_input=store_inp, store_output=store_oup, stop_forward=True)
        handle = module.register_forward_hook(data_saver)
        cached = []
        with torch.no_grad():
            for batch in cali_data:
                try:
                    _ = model(batch.to(device))
                except StopForwardException:
                    pass
                if store_inp:
                    cached.append([inp.detach() for inp in data_saver.input_store])
                if store_oup:
                    cached.append(data_saver.output_store.detach())
        handle.remove()
        return cached


Then you can design the PTQ function like:


.. code-block:: python 
    :linenos:

    
    def PTQ(model, data, *args, **kwargs):
        ptq_model = deepcopy_graphmodule(model)
        # diable the original model's update
        model.eval() 
        # turn the original model into float
        disable_all(model)
        # turn the ptq model into quant 
        enable_quantization(ptq_model) 
        nodes = list(model.graph.nodes)
        modules = dict(model.named_modules())
        quant_modules = dict(ptq_model.named_modules())
        for node in nodes:
            if node.op == "call_module" and isinstance(modules[node.target], _PTQ_SUPPORT_TYPE):
                module = modules[node.target]
                quant_module = quant_modules[node.target]
                cached_oups = save_inp_oup_data(model, module, cali_data, 
                                                store_inp=False, store_oup=True)
                cached_inps = save_inp_oup_data(quant_model, quant_module, cali_data, 
                                                store_inp=True, store_oup=False)
                # this will update the quant_module's params
                do_your_calibration(quant_module, cached_inps, cached_oups) 
        return quant_model


