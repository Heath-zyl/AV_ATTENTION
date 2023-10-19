import sys
import os
import time
import ray


@ray.remote(num_gpus=2)
def use_gpu():
    print("hello")
    time.sleep(5)
    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))


def main():
    results = [use_gpu.remote() for i in range(1)]
    time.sleep(10)
    print(ray.get(results))

    print("Success!")
    sys.stdout.flush()


if __name__ == "__main__":
    ray.init("ray://172.16.10.29:32001")
    main()