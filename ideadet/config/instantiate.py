# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import logging
from collections import abc
from enum import Enum
from typing import Any, Callable, Dict, List, Union
from hydra.errors import InstantiationException
from omegaconf import OmegaConf

from ideadet.config.lazy import _convert_target_to_string, locate

logger = logging.getLogger(__name__)

__all__ = ["dump_dataclass", "instantiate"]

# --------------------------------------------------------
# References:
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/instantiate.py
# --------------------------------------------------------


class _Keys(str, Enum):
    """Special keys in configs used by instantiate."""

    TARGET = "_target_"
    RECURSIVE = "_recursive_"


def _is_target(x: Any) -> bool:
    if isinstance(x, dict):
        return _Keys.TARGET in x
    if OmegaConf.is_dict(x):
        return _Keys.TARGET in x
    return False


def _is_dict(cfg: Any) -> bool:
    return OmegaConf.is_dict(cfg) or isinstance(cfg, abc.Mapping)


def _is_list(cfg: Any) -> bool:
    return OmegaConf.is_list(cfg) or isinstance(cfg, list)


def dump_dataclass(obj: Any):
    """
    Dump a dataclass recursively into a dict that can be later instantiated.

    Args:
        obj: a dataclass object

    Returns:
        dict
    """
    assert dataclasses.is_dataclass(obj) and not isinstance(
        obj, type
    ), "dump_dataclass() requires an instance of a dataclass."
    ret = {"_target_": _convert_target_to_string(type(obj))}
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        if dataclasses.is_dataclass(v):
            v = dump_dataclass(v)
        if isinstance(v, (list, tuple)):
            v = [dump_dataclass(x) if dataclasses.is_dataclass(x) else x for x in v]
        ret[f.name] = v
    return ret


def _prepare_input_dict_or_list(d: Union[Dict[Any, Any], List[Any]]) -> Any:
    res: Any
    if isinstance(d, dict):
        res = {}
        for k, v in d.items():
            if k == "_target_":
                v = _convert_target_to_string(d["_target_"])
            elif isinstance(v, (dict, list)):
                v = _prepare_input_dict_or_list(v)
            res[k] = v
    elif isinstance(d, list):
        res = []
        for v in d:
            if isinstance(v, (list, dict)):
                v = _prepare_input_dict_or_list(v)
            res.append(v)
    else:
        assert False
    return res


def _resolve_target(target):
    if isinstance(target, str):
        try:
            target = locate(target)
        except Exception as e:
            msg = f"Error locating target '{target}', see chained exception above."
            raise InstantiationException(msg) from e

    if not callable(target):
        msg = f"Expected a callable target, got '{target}' of type '{type(target).__name__}'"
        raise InstantiationException(msg)
    return target


def _call_target(_target_: Callable[..., Any], kwargs: Dict[str, Any]):
    """Call target (type) with kwargs"""

    try:
        return _target_(**kwargs)
    except Exception as e:
        msg = f"Error in call to target '{_convert_target_to_string(_target_)}':\n{repr(e)}"
        raise InstantiationException(msg) from e


def instantiate(cfg, **kwargs: Any) -> Any:
    """
    Recursively instantiate objects defined in dictionaries by
    "_target_" and arguments.

    Args:
        cfg: a dict-like object with "_target_" that defines the caller, and
            other keys that define the arguments

    Returns:
        object instantiated by cfg
    """
    if cfg is None:
        return None

    if isinstance(cfg, (dict, list)):
        cfg = _prepare_input_dict_or_list(cfg)

    kwargs = _prepare_input_dict_or_list(kwargs)

    if _is_dict(cfg):
        if kwargs:
            cfg = OmegaConf.merge(cfg, kwargs)

        _recursive_ = kwargs.pop(_Keys.RECURSIVE, True)
        return instantiate_cfg(cfg, recursive=_recursive_)

    elif _is_list(cfg):
        _recursive_ = kwargs.pop(_Keys.RECURSIVE, True)
        return instantiate_cfg(cfg, recursive=_recursive_)
    else:
        return cfg  # return as-is if don't know what to do


def instantiate_cfg(cfg: Any, recursive: bool = True):
    if cfg is None:
        return cfg

    if _is_dict(cfg):
        recursive = cfg[_Keys.RECURSIVE] if _Keys.RECURSIVE in cfg else recursive

    if not isinstance(recursive, bool):
        msg = f"Instantiation: _recursive_ flag must be a bool, got {type(recursive)}"
        raise TypeError(msg)

    # If OmegaConf list, create new list of instances if recursive
    if OmegaConf.is_list(cfg):
        items = [instantiate_cfg(item, recursive=recursive) for item in cfg._iter_ex(resolve=True)]
        lst = OmegaConf.create(items, flags={"allow_objects": True})
        return lst

    elif isinstance(cfg, list):
        # Specialize for list, because many classes take
        # list[objects] as arguments, such as ResNet, DatasetMapper
        return [instantiate(item, recursive=recursive) for item in cfg]

    elif _is_dict(cfg):
        exclude_keys = set({"_target_", "_recursive_"})
        if _is_target(cfg):
            _target_ = instantiate(cfg.get(_Keys.TARGET))  # instantiate lazy target
            _target_ = _resolve_target(_target_)
            kwargs = {}
            for key, value in cfg.items():
                if key not in exclude_keys:
                    if recursive:
                        value = instantiate_cfg(value, recursive=recursive)
                    kwargs[key] = value
            return _call_target(_target_, kwargs)
        else:
            return cfg
    else:
        return cfg  # return as-is if don't know what to do
