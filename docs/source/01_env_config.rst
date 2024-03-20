Environment Configuration
=========================================


Download code from github
-------------------------------

.. code-block:: bash

    git clone https://github.com/sophgo/sophgo-mq.git

After the code has been downloaded, we recommend compiling it using our Docker environment. 
For Docker configuration, please refer to the subsequent sections.



Basic Environment Configuration
---------------------------------
Download the required image from DockerHub :


.. code-block:: shell

   docker pull bernardxu6034/sophgo-mq:latest


If you are using docker for the first time, you can execute the following commands to install and configure it (only for the first time):


.. code-block:: shell

    sudo apt install docker.io
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo groupadd docker
    sudo usermod -aG docker $USER
    newgrp docker

Make sure the installation package is in the current directory, and then create a container in the current directory as follows:


.. code-block:: shell

    docker run --privileged --name myname -v $PWD:/workspace -it bernardxu6034/sophgo-mq:latest
    # "myname" is just an example, you can use any name you want


In the running Docker container, compile sophgo-mq using the following command:

.. code-block:: shell

    cd sophgo-mq
    python setup.py install
