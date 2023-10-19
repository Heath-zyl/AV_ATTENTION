
#!/bin/bash

cluster_name="crystal"

script_dir=$(cd $(dirname $0);pwd)
cluster_dir=$script_dir/../
ray_dir=$cluster_dir/deploy/ray
kube_config="$cluster_dir/kube.config"

# kubectl --kubeconfig=${kube_config}  delete ServiceAccount      ray-operator-serviceaccount
# kubectl --kubeconfig=${kube_config}  delete ClusterRole         ray-operator-clusterrole
# kubectl --kubeconfig=${kube_config}  delete ClusterRoleBinding  ray-operator-clusterrolebinding

helm --kubeconfig=$kube_config -n ray upgrade $cluster_name $ray_dir