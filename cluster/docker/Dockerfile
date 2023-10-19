FROM registry.aibee.cn/face-vision/torchpp:1.7.1
RUN mkdir -p /workspace
WORKDIR /workspace
COPY ./model /workspace/model
COPY ./scripts /workspace/scripts
COPY ./inference.py /workspace/inference.py
COPY ./train.py /workspace/train.py
COPY ./master_ops.py /workspace/master_ops.py
COPY ./data /workspace/data
COPY ./master_ops.py /workspace
COPY ./inference.py /workspace
COPY ./train.py /workspace
COPY ./cluster.conf /workspace
COPY ./README.md /workspace
