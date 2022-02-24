FROM xilinx/vitis-ai:1.4.1.978

RUN cd /home \
    && git clone https://github.com/ModelTC/MQBench.git \
    && cd MQBench \
    && sed '1,2d' requirements.txt > requirements.txt 

RUN source /opt/vitis_ai/conda/etc/profile.d/conda.sh \
    && source /home/vitis-ai-user/.bashrc \
    && conda activate vitis-ai-pytorch \
    && cd /home/MQBench \
    && pip install onnx -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install . 

RUN echo "conda activate vitis-ai-pytorch" >> /home/vitis-ai-user/.bashrc