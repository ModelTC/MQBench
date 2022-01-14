FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# Due to bad internet issue
RUN rm /etc/apt/sources.list.d/*

# prepare build dir tmp
RUN chmod 1777 /tmp \
    && mkdir /scratch \
    && chmod 1777 /scratch 

# install build needed tools 
RUN apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
        apt-transport-https \
        autoconf \
        automake \
        bc \
        build-essential \
        bzip2 \
        ca-certificates \
        curl \
        g++ \
        gdb \
        git \
        gnupg \
        locales \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libjson-c-dev \
        libjsoncpp-dev \
        libssl-dev \
        libtool \
        libunwind-dev \
        make \
        openssh-client \
        openssl \
        # python3 \
        python3-dev \
        # python3-minimal \
        # python3-numpy \
        # python3-opencv \
        python3-pip \
        # python3-setuptools \
        # python3-venv \
        software-properties-common \
        sudo \
        tree \
        unzip \
        vim \
        wget \
        yasm \
        zstd \
        libtool 

# set python 
ENV PYTHONPATH=/opt/python:$PYTHONPATH
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

ENV PYENV_ROOT=/opt/pyenv \
    PATH=/opt/pyenv/shims:/opt/pyenv/bin:$PATH

RUN git clone --depth 1 https://github.com/pyenv/pyenv.git /opt/pyenv \
    && eval "$(pyenv init -)" \
    && env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.6.4 \
    && pyenv global 3.6.4

RUN pip install --upgrade pip
RUN pip install torch==1.8.1 torchvision flake8 scipy pytest


# RUN mkdir /opt/hub && wget https://github.com/pytorch/vision/archive/master.zip -P /opt/hub && unzip /opt/hub/master.zip -d /opt/hub && mv /opt/hub/vision-master/ /opt/hub/pytorch_vision_master
ENV TORCH_HOME=/opt/

RUN apt-get install -y \
        libavcodec-dev \
        libavformat-dev \
        libeigen3-dev \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer1.0-dev \
        libgtest-dev \
        libgtk-3-dev \
        libgtk2.0-dev \
        libhdf5-dev \
        libjpeg-dev \
        libopenexr-dev \
        libpng-dev \
        libswscale-dev \
        libtiff-dev \
        libwebp-dev \
        # opencl-clhpp-headers \
        # opencl-headers \
        # pocl-opencl-icd \
        rpm \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get install -y \
        gcc-8 \
        g++-8 \
        gcc-9 \
        g++-9 \
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
    && apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' \
    && apt-get update -y \
    && apt-get install -y \
        cmake=3.16.0-0kitware1 \
        cmake-data=3.16.0-0kitware1 \
        kitware-archive-keyring \
    && apt-get install -y ffmpeg 

# set locales 
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
    && echo "LC_ALL=en_US.UTF-8" >> /etc/environment \ 
    && echo "LANG=en_US.UTF-8" > /etc/locale.conf \
    && locale-gen en_US.UTF-8 \
    && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8 \
    && apt-get install -y dialog \
    && dpkg-reconfigure --frontend noninteractive locales 


# build gtest 
RUN cd /usr/src/gtest \
    && mkdir -p build \
    && cd build \
    && cmake .. \
    && make \
    && make install

# set glob 
RUN cd /tmp \
    && wget --progress=dot:mega -O glog.0.4.0.tar.gz https://codeload.github.com/google/glog/tar.gz/v0.4.0 \
    && tar -xvf glog.0.4.0.tar.gz \
    && cd glog-0.4.0 \
    && ./autogen.sh \
    && mkdir build \
    && cd build \
    && cmake -DBUILD_SHARED_LIBS=ON .. \
    && make -j 4 \
    && make install \
    && rm -fr /tmp/*


# set protobuf 
RUN /tmp; wget --progress=dot:mega https://codeload.github.com/google/protobuf/zip/v3.4.0 \
    && unzip v3.4.0 \
    && cd protobuf-3.4.0 \
    && ./autogen.sh \
    && ./configure \
    && make -j 4 \
    && make install \
    && ldconfig \
    && rm -fr /tmp/*

# set gflag
RUN cd /tmp; wget --progress=dot:mega https://github.com/gflags/gflags/archive/v2.2.2.tar.gz \
    && tar xvf v2.2.2.tar.gz \
    && cd gflags-2.2.2 \
    && mkdir build \
    && cd build \
    && cmake -DBUILD_SHARED_LIBS=ON .. \
    && make -j 4 \
    && make install \
    && rm -fr /tmp/*

# set pybind
RUN cd /tmp; git clone https://github.com/pybind/pybind11.git \
    && cd pybind11 \
    && git checkout v2.5.0 \
    && mkdir build \
    && cd build \
    && cmake -DPYBIND11_TEST=OFF .. \
    && make \
    && make install \
    && rm -fr /tmp/* \
    && chmod 777 /usr/lib/python3/dist-packages

# set xir 
RUN cd /tmp \
    && wget -O libunilog.deb https://www.xilinx.com/bin/public/openDownload?filename=libunilog_1.4.1-r82_amd64.deb \
    && wget -O libtarget-factory.deb https://www.xilinx.com/bin/public/openDownload?filename=libtarget-factory_1.4.1-r85_amd64.deb \
    && wget -O libxir.deb https://www.xilinx.com/bin/public/openDownload?filename=libxir_1.4.1-r91_amd64.deb \
    && wget -O libvart.deb https://www.xilinx.com/bin/public/openDownload?filename=libvart_1.4.1-r130_amd64.deb \
    && wget -O libvitis_ai_library.deb https://www.xilinx.com/bin/public/openDownload?filename=libvitis_ai_library_1.4.1-r114_amd64.deb \
    && wget -O librt-engine.deb https://www.xilinx.com/bin/public/openDownload?filename=librt-engine_1.4.1-r195_amd64.deb \
    && wget -O aks.deb https://www.xilinx.com/bin/public/openDownload?filename=aks_1.4.1-r78_amd64.deb \
    && apt-get install -y --no-install-recommends /tmp/*.deb \
    && rm -rf /tmp/* \
    && ldconfig   

# install mqbench requirement
RUN pip install urllib3 onnx 

# set xir path 
ENV PYTHONPATH=/usr/lib/python3/dist-packages/:$PYTHONPATH

# install mqbench 
RUN cd /root/ \
    && git clone https://github.com/ModelTC/MQBench.git \
    && cd MQBench \
    && python setup.py develop 

