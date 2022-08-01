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


import argparse
import sys


def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by libai users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:

Run on single machine:
    $ python3 -m oneflow.distributed.launch \
    --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 {sys.argv[0]} \
    --config-file cfg.yaml

Change some config options:
    $ python3 -m oneflow.distributed.launch \
    --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 {sys.argv[0]} \
    --config-file cfg.yaml train.load_weight=/path/to/weight.pth optim.lr=0.001

Run on multiple machines:
    (machine0)$ python3 -m oneflow.distributed.launch \
    --nproc_per_node 8 --nnodes 2 --node_rank 0 --master_addr <URL> {sys.argv[0]} \
    --config-file cfg.yaml

    $ python3 -m oneflow.distributed.launch \
    --nproc_per_node 8 --nnodes 2 --node_rank 1 --master_addr <URL> {sys.argv[0]} \
    --config-file cfg.yaml

""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run several batches of train, eval and test to find any bugs, "
        "(ie: a sort of unit test)",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "path.key value" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser
