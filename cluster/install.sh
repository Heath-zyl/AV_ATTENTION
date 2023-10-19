#!/bin/bash
pip uninstall -y torchx
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
mkdir -p /root/.kube && cp kube.config /root/.kube/config