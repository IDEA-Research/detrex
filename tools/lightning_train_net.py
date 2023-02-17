
#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# Lightning Trainer should be considered beta at this point
# We have confirmed that training and validation run correctly and produce correct results
# Depending on how you launch the trainer, there are issues with processes terminating correctly
# This module is still dependent on D2 logging, but could be transferred to use Lightning logging

import logging
import os
import time
import weakref
from collections import OrderedDict
from typing import Any, Dict, List
import pytorch_lightning as pl  # type: ignore
from pytorch_lightning import LightningDataModule, LightningModule

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
)
from detectron2.evaluation import print_csv_format
from detectron2.evaluation.testing import flatten_results_dict
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import setup_logger
from detectron2.solver.lr_scheduler import LRMultiplier


logger = logging.getLogger("detectron2")


class TrainingModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        self.cfg = cfg
        self.storage: EventStorage = None
        self.model = instantiate(cfg.model)

        self.start_iter = 0
        self.max_iter = cfg.train.max_iter


    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["iteration"] = self.storage.iter

    def on_load_checkpoint(self, checkpointed_state: Dict[str, Any]) -> None:
        self.start_iter = checkpointed_state["iteration"]
        self.storage.iter = self.start_iter

    def setup(self, stage: str):
        self.checkpointer = DetectionCheckpointer(
            self.model,
            self.cfg.train.output_dir
        )
        if self.cfg.train.init_checkpoint:
            logger.info(f"Load model weights from checkpoint: {self.cfg.train.init_checkpoint}.")
            self.checkpointer.load(self.cfg.train.init_checkpoint)
        
        self.iteration_timer = hooks.IterationTimer()
        self.iteration_timer.before_train()
        self.data_start = time.perf_counter()
        self.writers = None

    def training_step(self, batch, batch_idx):
        data_time = time.perf_counter() - self.data_start
        # Need to manually enter/exit since trainer may launch processes
        # This ideally belongs in setup, but setup seems to run before processes are spawned
        if self.storage is None:
            self.storage = EventStorage(0)
            self.storage.__enter__()
            self.iteration_timer.trainer = weakref.proxy(self)
            self.iteration_timer.before_step()
            self.writers = (
                default_writers(self.cfg.train.output_dir, self.max_iter)
                if comm.is_main_process()
                else {}
            )

        loss_dict = self.model(batch)
        SimpleTrainer.write_metrics(loss_dict, data_time)

        opt = self.optimizers()
        self.storage.put_scalar(
            "lr", opt.param_groups[self._best_param_group_id]["lr"], smoothing_hint=False
        )
        self.iteration_timer.after_step()
        self.storage.step()
        # A little odd to put before step here, but it's the best way to get a proper timing
        self.iteration_timer.before_step()

        if self.storage.iter % 20 == 0:
            for writer in self.writers:
                writer.write()
        return sum(loss_dict.values())
    
    def training_step_end(self, training_step_outpus):
        self.data_start = time.perf_counter()
        return training_step_outpus
    
    def training_epoch_end(self, training_step_outputs):
        self.iteration_timer.after_train()
        if comm.is_main_process():
            self.checkpointer.save("model_final")
        for writer in self.writers:
            writer.write()
            writer.close()
        self.storage.__exit__(None, None, None)

    def _process_dataset_evaluation_results(self) -> OrderedDict:
        results = self._evaluators[0].evaluate()
        if comm.is_main_process():
            print_csv_format(results)

        if len(results) == 1:
            results = list(results.values())[0]
        return results
    
    def _reset_dataset_evaluators(self):
        self._evaluators = []
        evaluator = instantiate(self.cfg.dataloader.evaluator)
        evaluator.reset()
        self._evaluators.append(evaluator)

    def on_validation_epoch_start(self, _outputs):
        self._reset_dataset_evaluators()

    def validation_epoch_end(self, _outputs):
        results = self._process_dataset_evaluation_results(_outputs)

        flattened_results = flatten_results_dict(results)
        for k, v in flattened_results.items():
            try:
                v = float(v)
            except Exception as e:
                raise ValueError(
                    "[EvalHook] eval_function should return a nested dict of float. "
                    "Got '{}: {}' instead.".format(k, v)
                ) from e
        self.storage.put_scalars(**flattened_results, smoothing_hint=False)

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        if not isinstance(batch, List):
            batch = [batch]
        outputs = self.model(batch)
        self._evaluators[dataloader_idx].process(batch, outputs)

    def configure_optimizers(self):
        self.cfg.optimizer.params.model = self.model
        optimizer = instantiate(self.cfg.optimizer)
        self._best_param_group_id = hooks.LRScheduler.get_best_param_group_id(optimizer)
        multiplier = instantiate(self.cfg.lr_multiplier)
        scheduler = LRMultiplier(optimizer, multiplier, max_iter=self.cfg.train.max_iter)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    

class DataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def train_dataloader(self):
        return instantiate(self.cfg.dataloader.train)

    def val_dataloader(self):
        dataloaders = []
        dataloaders.append(instantiate(self.cfg.dataloader.test))
        return dataloaders
    

def main(args):
    cfg = setup(args)
    train(cfg, args)


def train(cfg, args):
    trainer_params = {
        # training loop is bounded by max steps, use a large max_epochs to make
        # sure max_steps is met first
        "max_epochs": 10**8,
        "max_steps": cfg.train.max_iter,
        "val_check_interval": cfg.train.eval_period if cfg.train.eval_period > 0 else 10**8,
        "num_nodes": args.num_machines,
        "devices": args.num_gpus,
        "num_sanity_val_steps": 0,
        "accelerator": "gpu",
    }
    if cfg.train.amp.enabled:
        trainer_params["precision"] = 16

    last_checkpoint = os.path.join(cfg.train.output_dir, "last.ckpt")
    if args.resume:
        # resume training from checkpoint
        trainer_params["resume_from_checkpoint"] = last_checkpoint
        logger.info(f"Resuming training from checkpoint: {last_checkpoint}.")

    trainer = pl.Trainer(**trainer_params)
    logger.info(f"start to train with {args.num_machines} nodes and {args.num_gpus} GPUs")
    module = TrainingModule(cfg)
    data_module = DataModule(cfg)
    if args.eval_only:
        logger.info("Running inference")
        trainer.validate(module, data_module)
    else:
        logger.info("Running training")
        trainer.fit(module, data_module)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)
    return cfg


if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()
    logger.info("Command Line Args:", args)
    main(args)