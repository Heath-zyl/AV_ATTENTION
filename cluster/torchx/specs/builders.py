# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
import argparse
import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from torchx.specs.api import BindMount, MountType, VolumeMount, Toleration, NodeSelector
from .api import AppDef, DeviceMount


def make_app_handle(scheduler_backend: str, session_name: str, app_id: str) -> str:
    return f"{scheduler_backend}://{session_name}/{app_id}"


_MOUNT_OPT_MAP: Mapping[str, str] = {
    "type": "type",
    "destination": "dst",
    "dst": "dst",
    "target": "dst",
    "read_only": "readonly",
    "readonly": "readonly",
    "source": "src",
    "src": "src",
    "perm": "perm",
}


def parse_mounts(opts: List[str]) -> List[Union[BindMount, VolumeMount, DeviceMount]]:
    """
    parse_mounts parses a list of options into typed mounts following a similar
    format to Dockers bind mount.

    Multiple mounts can be specified in the same list. ``type`` must be
    specified first in each.

    Ex:
        type=bind,src=/host,dst=/container,readonly,[type=bind,src=...,dst=...]

    Supported types:
        BindMount: type=bind,src=<host path>,dst=<container path>[,readonly]
        VolumeMount: type=volume,src=<name/id>,dst=<container path>[,readonly]
        DeviceMount: type=device,src=/dev/<dev>[,dst=<container path>][,perm=rwm]
    """
    mount_opts = []
    cur = {}
    for opt in opts:
        key, _, val = opt.partition("=")
        if key not in _MOUNT_OPT_MAP:
            raise KeyError(
                f"unknown mount option {key}, must be one of {list(_MOUNT_OPT_MAP.keys())}"
            )
        key = _MOUNT_OPT_MAP[key]
        if key == "type":
            cur = {}
            mount_opts.append(cur)
        elif len(mount_opts) == 0:
            raise KeyError("type must be specified first")
        cur[key] = val

    mounts = []
    for opts in mount_opts:
        typ = opts.get("type")
        if typ == MountType.BIND:
            mounts.append(
                BindMount(
                    src_path=opts["src"],
                    dst_path=opts["dst"],
                    read_only="readonly" in opts,
                )
            )
        elif typ == MountType.VOLUME:
            mounts.append(
                VolumeMount(
                    src=opts["src"], dst_path=opts["dst"], read_only="readonly" in opts
                )
            )
        elif typ == MountType.DEVICE:
            src = opts["src"]
            dst = opts.get("dst", src)
            perm = opts.get("perm", "rwm")
            for c in perm:
                if c not in "rwm":
                    raise ValueError(
                        f"{c} is not a valid permission flags must one of r,w,m"
                    )
            mounts.append(DeviceMount(src_path=src, dst_path=dst, permissions=perm))
        else:
            valid = list(str(item.value) for item in MountType)
            raise ValueError(f"invalid mount type {repr(typ)}, must be one of {valid}")
    return mounts


def parse_tolerations(opts: str) -> List[Toleration]:
    """
    parse_mounts parses a list of options into typed mounts following a similar
    format to Dockers bind mount.

    Multiple mounts can be specified in the same list. ``type`` must be
    specified first in each.

    Ex:
        key=value:Policy,[key=value:Policy]
        key:Policy,[key:Policy]
        key,[key]

    """
    tolerations = []
    
    opts = opts.split(",")
    for opt in opts:
        if len(opt) < 1:
            continue

        arr = re.split('=|:',opt)
        
        key = arr[0]
        if len(arr) == 1:
            value = None
            effect = "NoSchedule"
        elif len(arr) == 2:
            value = None
            effect = arr[1]
        else:
            value = arr[1]
            effect = arr[2]
    
        toleration = Toleration(key=key, value=value, effect=effect, operator="Equal")
        tolerations.append(toleration)
    
    return tolerations

def parse_node_selectors(opts: str) -> List[NodeSelector]:
    """
    parse_mounts parses a list of options into typed mounts following a similar
    format to Dockers bind mount.

    Multiple mounts can be specified in the same list. ``type`` must be
    specified first in each.

    Ex:
        key:value,key:value

    """
    selectors = []
    if len(opts) < 1 or opts is None:
        return selectors
    
    opts = opts.split(",")
    for opt in opts:
        arr = opt.split(":")
        if len(arr) < 2:
            continue
        
        selector = NodeSelector(key=arr[0], value=arr[1])
        selectors.append(selector)
    
    return selectors

def parse_env(opts: str) -> Dict[str, str]:
    """
    parse_mounts parses a list of options into typed mounts following a similar
    format to Dockers bind mount.

    Multiple mounts can be specified in the same list. ``type`` must be
    specified first in each.

    Ex:
        key:value,key:value

    """
    envs = {}
    opts = opts.split(",")

    for opt in opts:
        arr = opt.split("=")
        if len(arr) < 2:
            continue

        envs[arr[0]] = arr[1]
    
    return envs