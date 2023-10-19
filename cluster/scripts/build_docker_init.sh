#!/bin/bash
tag=1.1
docker_addr=registry.aibee.cn/aibee/busybox_waitservice:$tag
docker build -f docker/Dockerfile.init . -t $docker_addr
docker push $docker_addr
