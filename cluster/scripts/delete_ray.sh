
#!/bin/bash

cluster_name="crystal"
script_dir=$(cd $(dirname $0);pwd)
cluster_dir=$script_dir/../
ray_dir=$cluster_dir/deploy/ray

kube_config="/home/yongli/crystal/cluster/kube.config"

# kubectl --kubeconfig=${kube_config}  delete ServiceAccount      ray-operator-serviceaccount
# kubectl --kubeconfig=${kube_config}  delete ClusterRole         ray-operator-clusterrole
# kubectl --kubeconfig=${kube_config}  delete ClusterRoleBinding  ray-operator-clusterrolebinding

# 删除namespace=default的资源，比如ServiceAccount/ClusterRole/ClusterRoleBinding
helm --kubeconfig=$kube_config  uninstall $cluster_name 
helm --kubeconfig=$kube_config -n ray uninstall $cluster_name 
