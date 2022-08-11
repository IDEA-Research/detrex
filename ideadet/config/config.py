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


import os
import pkg_resources
from omegaconf import OmegaConf
from detectron2.config import LazyConfig


def try_get_key(cfg, *keys, default=None):
    """
    Try select keys from cfg until the first key that exists. Otherwise return default.
    """
    for k in keys:
        none = object()
        p = OmegaConf.select(cfg, k, default=none)
        if p is not none:
            return p
    return default


def get_config(config_path):
    """
    Returns a config object from a config_path.
    Args:
        config_path (str): config file name relative to ideadet's "configs/"
            directory, e.g., "common/models/bert.py"
    Returns:
        omegaconf.DictConfig: a config object
    """
    cfg_file = pkg_resources.resource_filename("ideadet.config", os.path.join("configs", config_path))
    if not os.path.exists(cfg_file):
        raise RuntimeError("{} not available in IDEADet configs!".format(config_path))
    cfg = LazyConfig.load(cfg_file)
    return cfg