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

import wandb
from PIL import Image
from omegaconf import OmegaConf

from detectron2.utils.events import EventWriter, get_event_storage


class WandbWriter(EventWriter):
    """
    Write all scalars to a wandb file.
    """

    def __init__(self, cfg, window_size: int = 20, **kwargs):
        """
        Args:
            log_dir (str): the directory to save the output events
            window_size (int): the scalars will be median-smoothed by this window size

            kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
        """
        self._window_size = window_size

        self._writer = wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.train.wandb.params,
        )
        self._last_write = -1
    
    def write(self):
        storage = get_event_storage()
        new_last_write = self._last_write
        for k, (v, iter) in storage.latest_with_smoothing_hint(self._window_size).items():
            if iter > self._last_write:
                self._writer.log({k: v}, step=iter)
                new_last_write = max(new_last_write, iter)
        self._last_write = new_last_write

        # visualize training samples
        if len(storage._vis_data) >= 1:
            for img_name, img, step_num in storage._vis_data:
                log_img = Image.fromarray(img.transpose(1, 2, 0))  # convert to (h, w, 3) PIL.Image
                log_img = wandb.Image(log_img, caption=img_name)
                self._writer.log({img_name: [log_img]})
            # Storage stores all image data and rely on this writer to clear them.
            # As a result it assumes only one writer will use its image data.
            # An alternative design is to let storage store limited recent
            # data (e.g. only the most recent image) that all writers can access.
            # In that case a writer may not see all image data if its period is long.
            storage.clear_images()
    
    def close(self):
        if hasattr(self, "_writer"):
            self._writer.finish()
