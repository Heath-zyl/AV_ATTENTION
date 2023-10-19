import time
import signal
import sys
import subprocess
import os
from argparse import ArgumentParser, REMAINDER
import configargparse
from typing import Optional, IO, List, Any
from .scheduler import get_scheduler

node_local_rank_stdout_filename = "node_{}_local_rank_{}_stdout"
node_local_rank_stderr_filename = "node_{}_local_rank_{}_stderr"

def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = configargparse.ArgParser(ignore_unknown_config_file_keys=True)
    parser.add('-c', '--config', required=True, 
                        is_config_file=True, help='config file path')
    parser.add("--scheduler", type=str, required=True,
                        help="scheduler to submit job ")
    parser.add("--ldap", type=str, required=True,
                        help="ldap of submitter ")
    parser.add("--jobname", type=str, required=True,
                        help="job name of in cluster ")
    parser.add("--namespace", type=str, required=True,
                        help="namespace to submit job")
    parser.add("--comment", type=str, required=True,
                        help="ldap of submitter ")
    # Optional arguments for the launch helper
    parser.add("--nnodes", type=int, required=True,
                        help="The number of nodes to use for distributed ")
    parser.add("--nproc_per_node", type=int, required=True,
                        help="The number of processes to launch on each node, "
                             "for GPU training, this is recommended to be set "
                             "to the number of GPUs in your system so that "
                             "each process can be bound to a single GPU.")
    parser.add("--ncpu_per_node", type=int, required=True,
                    help="The number of cpu to request from cluster, "
                        "training")
    parser.add("--mbmem_per_node", type=int, required=True,
                    help="memory of cpu to request from cluster, the unit is MB"
                        "training")
    parser.add("--ngpu_per_node", type=int, required=True,
                    help="The number of gpus on each node, "
                            "for GPU training, this is recommended to be set "
                            "to the number of GPUs in your system so that "
                            "each process can be bound to a single GPU.")
    parser.add("--mounts", type=str, required=True,
                    help="mount local dir to container"
                        "training")
    parser.add("--node_selector", type=str, required=True,
                    help="node selector for deploy"
                        "spec")
    parser.add("--tolerations", type=str, required=True,
                    help="node selector for deploy"
                        "spec")

    parser.add("--env", type=str, required=True,
                    help="env for deploy"
                        "spec")
    parser.add("--dir_list", type=str, required=True,
                    help="dir list for building docker image"
                        "training")
    parser.add("--base_image", type=str, required=True,
                    help="dir list for building docker image")
    parser.add("--docker_file", type=str, required=True,
                    help="dir list for building docker image")
    parser.add("--yaml_template", type=str, required=True,
                    help="dir list for building docker image")

    # for host env 
    parser.add("--kube_cmd", type=str, required=True,
                    help="dir list for building docker image")
    parser.add("--kube_conf", type=str, required=True,
                    help="dir list for building docker image")
    parser.add("--use_env", default=False, action="store_true",
                        help="Use environment variable to pass "
                             "'local rank'. For legacy reasons, the default value is False. "
                             "If set to True, the script will not pass "
                             "--local_rank as argument, and will instead set LOCAL_RANK.")
    parser.add("-m", "--module", default=False, action="store_true",
                        help="Changes each process to interpret the launch script "
                             "as a python module, executing with the same behavior as"
                             "'python -m'.")
    parser.add("--no_python", default=False, action="store_true",
                        help="Do not prepend the training script with \"python\" - just exec "
                             "it directly. Useful when the script is not a Python script.")
    # positional
    parser.add("training_script", type=str,
                        help="The full path to the single GPU training "
                             "program/script to be launched in parallel, "
                             "followed by all the arguments for the "
                             "training script")

    # rest from the training program
    parser.add('training_script_args',  nargs='+')
    return parser, parser.parse_args()

def main():
    # print(sys.argv[1:])
    p, args = parse_args()

    p,options = parse_args()
    print(options)
    print("----------")
    print(p.format_values()) 
    print("----------")

    processes: List[Any] = []
    subprocess_file_handles = []

    # spawn the processes
    with_python = not args.no_python
    cmd = []
    if with_python:
        cmd = [sys.executable, "-u"]
        if args.module:
            cmd.append("-m")
    else:
        if not args.use_env:
            raise ValueError("When using the '--no_python' flag, you must also set the '--use_env' flag.")
        if args.module:
            raise ValueError("Don't use both the '--no_python' flag and the '--module' flag at the same time.")

    cmd.extend(args.training_script_args)
    
    scheduler = get_scheduler(args.scheduler, args)
    scheduler.submit()

if __name__ == "__main__":
    main()