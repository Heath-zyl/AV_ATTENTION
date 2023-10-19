#!/bin/bash

current_dir="$(dirname "$(readlink -f "$0")")"
cluster_dir=${current_dir}/../

echo "cluster_dir: ${cluster_dir}"

# if [ -n "$(git status -s)" ];then
#     echo "git repo is not clean, may be there is no commited file!"
#     exit 1
# fi

# commit=$(git rev-parse --short HEAD)

tag=ray1.13.0-cluster1.1
docker_addr=registry.aibee.cn/aibee/crystal/lab:0.3-$tag
docker build -f $cluster_dir/docker/Dockerfile.crystal . -t $docker_addr
docker push $docker_addr