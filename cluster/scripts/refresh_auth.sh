
#!/bin/bash

cluster_name="crystal"
script_dir=$(cd $(dirname $0);pwd)
cluster_dir=$script_dir/../
ray_dir=$cluster_dir/deploy/ray


kubectl delete -f $cluster_dir/deploy/metrics-server/auth-reader.yaml.j2
kubectl apply -f $cluster_dir/deploy/metrics-server/auth-reader.yaml.j2

kubectl delete -f $cluster_dir/deploy/metrics-server/auth-delegator.yaml.j2
kubectl apply -f $cluster_dir/deploy/metrics-server/auth-delegator.yaml.j2
