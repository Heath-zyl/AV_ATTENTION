#!/usr/bin/env bash
DOCKER_ADDRESS=registry.aibee.cn/aibee/pytorch1.7.1-py38-cuda11.0-cudnn8-mmdetection

docker pull $DOCKER_ADDRESS

nvidia-docker run --shm-size=100gb -it -d \
    --network=host \
    --name av_attention \
    -e COLUMNS=`tput cols` \
    -e LINES=`tput lines` \
    -v /etc/localtime:/etc/localtime:ro \
    -v /ssd:/ssd \
    -v /mnt:/mnt \
    -v /root:/root \
    -v /var:/var \
    -v /etc:/root_etc \
    -v $PWD:/workspace \
    -v /ssd:/ssd \
    -v /home:/home \
    -v /face:/face \
    -v /training:/training \
    -p 12355:12355 \
    $DOCKER_ADDRESS \
    bash
