Setup MQBench
=============

**1**. **Install**

MQBench only depend on PyTorch 1.8.1, following `pytorch.org <http://pytorch.org/>`_ or use requirements file to install.

.. code-block:: shell
    :linenos:

    cd /path_of_mqbench              # change dir to MQBench
    pip install -r requirements.txt  # install MQBench dependencies
    python setup.py install          # install MQBench

**2**. **Validate**

Validate by executing following with no errors.

.. code-block:: shell
    :linenos:

    python -c 'import mqbench'

**3**. **Uninstall**

Remove MQBench by executing following shell script.

.. code-block:: shell
    :linenos:

    pip uninstall mqbench

You have done setup of MQBench, check :doc:`quick_start_deploy` or check :doc:`quick_start_academic` to get started with MQBench.