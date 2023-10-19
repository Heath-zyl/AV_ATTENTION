
import sys
import time
import ray

""" Run this script locally to execute a Ray program on your Ray cluster on
Kubernetes.

Before running this script, you must port-forward from the local host to
the relevant Kubernetes head service e.g.
kubectl -n ray port-forward service/example-cluster-ray-head 10001:10001.

Set the constant LOCAL_PORT below to the local port being forwarded.
"""

def main():
    print("ray clsuter resources:")
    print(ray.cluster_resources())
    print("--------------------")


    print("ray available resources:")
    print(ray.available_resources())
    print("--------------------")

    print("ray runtime context:")
    print(ray.get_runtime_context().get())


if __name__ == "__main__":
    ray.init("ray://172.16.10.29:32001")
    main()

