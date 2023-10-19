#!/usr/bin/env bash
DOCKER_ADDRESS=registry.aibee.cn/face-vision/torchpp:1.7.1

docker pull $DOCKER_ADDRESS

nvidia-docker run --shm-size=100gb -it -d \
    --network=host \
    --name av_attention \
    -e COLUMNS=`tput cols` \
    -e LINES=`tput lines` \
    -v /etc/localtime:/etc/localtime:ro \
    -v /ssd:/ssd \
    -v /mnt:/mnt \
    -v /var:/var \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /etc:/root_etc \
    -v $PWD:/workspace \
    -v /ssd:/ssd \
    -v /face:/face \
    -v /training:/training \
    -p 12345:12345 \
    $DOCKER_ADDRESS \
    bash -c 'export PATH=$PATH:/workspace/cluster/bin >> ~/.bashrc && bash'