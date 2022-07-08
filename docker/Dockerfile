FROM xilinx/vitis-ai:1.4.1.978

RUN cd /home \
    && git clone https://github.com/ModelTC/MQBench.git

RUN source /opt/vitis_ai/conda/etc/profile.d/conda.sh \
    && source /home/vitis-ai-user/.bashrc \
    && conda activate vitis-ai-pytorch \
    && cd /home/MQBench \
    && pip install onnx==1.8.0 \
    && cp mqbench/deploy/convert_xir.py /home/

RUN echo "conda activate vitis-ai-pytorch" >> /home/vitis-ai-user/.bashrc

RUN echo "echo usage: python /home/convert_xir.py [-h] -Q QMODEL -C CMODEL -N NAME" >> /home/vitis-ai-user/.bashrc