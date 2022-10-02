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
# ------------------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------------------------------
# Modified from:
# https://github.com/facebookresearch/detr/blob/main/d2/converter.py
# ------------------------------------------------------------------------------------------------

import argparse
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser("detrex deformable-detr model converter")

    parser.add_argument(
        "--source_model", default="", type=str, help="Path or url to the DETR model to convert"
    )
    parser.add_argument(
        "--output_model", default="", type=str, help="Path where to save the converted model"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # D2 expects contiguous classes, so we need to remap the 92 classes from DETR
    # fmt: off
    coco_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51,
                52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
                78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    # fmt: on

    coco_idx = np.array(coco_idx)

    if args.source_model.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.source_model, map_location="cpu", check_hash=True
        )
    else:
        checkpoint = torch.load(args.source_model, map_location="cpu")
    model_to_convert = checkpoint["model"]

    model_converted = {}
    for k in model_to_convert.keys():
        old_k = k
        if "backbone" in k:
            k = k.replace("backbone.0.body.", "")
            if "layer" not in k:
                k = "stem." + k
            for t in [1, 2, 3, 4]:
                k = k.replace(f"layer{t}", f"res{t + 1}")
            for t in [1, 2, 3]:
                k = k.replace(f"bn{t}", f"conv{t}.norm")
            k = k.replace("downsample.0", "shortcut")
            k = k.replace("downsample.1", "shortcut.norm")
            k = "backbone." + k

        # add input_proj converter
        if "input_proj" in k:
            if "input_proj.0.0" in k:
                k = k.replace("input_proj.0.0", "neck.convs.0.conv")
            if "input_proj.0.1" in k:
                k = k.replace("input_proj.0.1", "neck.convs.0.norm")
            if "input_proj.1.0" in k:
                k = k.replace("input_proj.1.0", "neck.convs.1.conv")
            if "input_proj.1.1" in k:
                k = k.replace("input_proj.1.1", "neck.convs.1.norm")
            if "input_proj.2.0" in k:
                k = k.replace("input_proj.2.0", "neck.convs.2.conv")
            if "input_proj.2.1" in k:
                k = k.replace("input_proj.2.1", "neck.convs.2.norm")
            if "input_proj.3.0" in k:
                k = k.replace("input_proj.3.0", "neck.extra_convs.0.conv")
            if "input_proj.3.1" in k:
                k = k.replace("input_proj.3.1", "neck.extra_convs.0.norm")

        # add new convert content
        if "encoder.layers" in k:
            if "self_attn" in k:
                k = k.replace("self_attn", "attentions.0")
            elif "linear1" in k:
                k = k.replace("linear1", "ffns.0.layers.0.0")
            elif "linear2" in k:
                k = k.replace("linear2", "ffns.0.layers.1")
            elif "norm1" in k:
                k = k.replace("norm1", "norms.0")
            elif "norm2" in k:
                k = k.replace("norm2", "norms.1")

        if "decoder" in k:
            if "linear1" in k:
                k = k.replace("linear1", "ffns.0.layers.0.0")
            elif "linear2" in k:
                k = k.replace("linear2", "ffns.0.layers.1")
            elif "norm1" in k:
                k = k.replace("norm1", "norms.1")
            elif "norm2" in k:
                k = k.replace("norm2", "norms.0")
            elif "norm3" in k:
                k = k.replace("norm3", "norms.2")
            elif "self_attn" in k:
                k = k.replace("self_attn", "attentions.0.attn")
            elif "cross_attn" in k:
                k = k.replace("cross_attn", "attentions.1")

        if "level_embed" in k:
            k = k.replace("level_embed", "level_embeds")

        if "query_embed" in k:
            k = k.replace("query_embed", "query_embedding")

        # k = "detr." + k
        print(old_k, "->", k)
        if "class_embed.6" in old_k:
            v = model_to_convert[old_k].detach()
            shape_old = v.shape
            v = v[:80]
            model_converted[k] = v
            print(
                "Head conversion: changing shape from {} to {}".format(
                    shape_old, model_converted[k].shape
                )
            )
            continue

        elif "class_embed" in old_k:
            v = model_to_convert[old_k].detach()
            if v.shape[0] == 91:
                shape_old = v.shape
                model_converted[k] = v[coco_idx]
                print(
                    "Head conversion: changing shape from {} to {}".format(
                        shape_old, model_converted[k].shape
                    )
                )
                continue
        model_converted[k] = model_to_convert[old_k].detach()

    model_to_save = {"model": model_converted}
    torch.save(model_to_save, args.output_model)


if __name__ == "__main__":
    main()
