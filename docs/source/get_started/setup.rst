Setup MQBench
============

**1**. **Installation**

MQBench only depend on PyTorch 1.8.1, following `pytorch.org <http://pytorch.org/>`_ or use requirements file to install.

.. code-block:: shell
    :linenos:

    cd /path_of_mqbench              # change dir to MQBench
    pip install -r requirements.txt  # install MQBench dependencies
    python setup.py install          # install MQBench

**2**. **Validation**

Validate by executing following with no errors.

.. code-block:: shell
    :linenos:

    python -c 'import mqbench'

You have done installation of MQBench, check :doc:`quick_start_deploy` or check :doc:`quick_start_academic` to get started with MQBench.