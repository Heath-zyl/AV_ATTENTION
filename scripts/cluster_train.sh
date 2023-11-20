#!/usr/bin/env bash
export PYTHONDONTWRITEBYTECODE=1
export PATH=/workspace/cluster/bin:$PATH
export PYTHONPATH=/workspace/cluster:$PYTHONPATH

CURRENT_TIME=`date "+%Y%m%d-%H%M%S"`


WORLD_SIZE=4
PORT=${PORT:-29502}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

python -m cluster.launch \
    --config /workspace/cluster.conf --ldap ylzhang --jobname avatn --namespace face  --nnodes $WORLD_SIZE \
    train.py -- \
    --workdir /face/ylzhang/tirl_workdir