import os
import sys
import weakref
import wandb
import torch
import torch.nn as nn
import torch.utils.data
import pprint
from packaging import version
from functools import partial
from pathlib import Path

if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter

from .defaults import create_ddp_model, worker_init_fn
from .hooks import HookBase, build_hooks
import pointcept.utils.comm as comm
from pointcept.utils.wandb_utils import is_wandb_active
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.optimizer import build_optimizer
from pointcept.utils.scheduler import build_scheduler
from pointcept.utils.events import EventStorage, ExceptionWriter
from pointcept.utils.registry import Registry

import warnings

warnings.filterwarnings(
    "ignore", "You are using `torch.load` with `weights_only=False`*."
)

TRAINERS = Registry("trainers")
AMP_DTYPE = dict(
    float16=torch.float16,
    bfloat16=torch.bfloat16,
)


def _resolve_wandb_run_name_and_tag(cfg):
    save_parts = Path(str(cfg.save_path)).parts
    if len(save_parts) >= 2:
        tag, name = save_parts[-2:]
        return f"{tag}/{name}", tag

    filename = getattr(cfg, "filename", None)
    if filename:
        config_path = Path(filename)
        run_name = config_path.stem or "run"
        tag = config_path.parent.name or "default"
        return run_name, tag

    return (save_parts[-1] if save_parts else "run"), "default"


class TrainerBase:
    def __init__(self) -> None:
        self.hooks = []
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = 0
        self.max_iter = 0
        self.comm_info = dict()
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage
        self.writer: SummaryWriter

    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

    def train(self):
        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def before_eval(self):
        for h in self.hooks:
            h.before_eval()

    def before_train(self):
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        for h in self.hooks:
            h.after_step()

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

    def after_train(self):
        # Sync GPU before running train hooks
        comm.synchronize()
        torch.cuda.empty_cache()
        for h in self.hooks:
            h.after_train()
        if comm.is_main_process():
            self.writer.close()


@TRAINERS.register_module("DefaultTrainer")
class Trainer(TrainerBase):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        self.best_metric_value = -torch.inf
        self.best_metrics = {}
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        if not self.cfg.test_only:
            self.logger.info("=> Building train dataset & dataloader ...")
            self.train_loader = self.build_train_loader()
            self.logger.info("=> Building val dataset & dataloader ...")
            self.val_loader = self.build_val_loader()
            self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
            self.optimizer = self.build_optimizer()
            self.scheduler = self.build_scheduler()
            self.scaler = self.build_scaler()
        else:
            self.train_loader = None
            self.val_loader = None
            self.optimizer = None
            self.scheduler = None
            self.scaler = None
        self.logger.info("=> Building hooks ...")
        pprint.pprint(self.cfg.hooks, indent=2)
        self.register_hooks(self.cfg.hooks)

    def train(self):
        with EventStorage() as self.storage, ExceptionWriter():
            # => before train
            if self.cfg.test_only:
                self.before_eval()
                self.logger.info(
                    ">>>>>>>>>>>>>>>> Test Only, Skip Training >>>>>>>>>>>>>>>>"
                )
            else:
                self.before_train()
                if self.cfg.set_detect_anomaly:
                    torch.autograd.set_detect_anomaly(True)
                self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
                for self.epoch in range(self.start_epoch, self.max_epoch):
                    # => before epoch
                    # TODO: optimize to iteration based
                    if comm.get_world_size() > 1:
                        self.train_loader.sampler.set_epoch(self.epoch)
                    self.model.train()
                    self.data_iterator = enumerate(self.train_loader)
                    self.before_epoch()
                    # => run_epoch
                    for (
                        self.comm_info["iter"],
                        self.comm_info["input_dict"],
                    ) in self.data_iterator:
                        # => before_step
                        self.before_step()
                        # => run_step
                        self.run_step()
                        # => after_step
                        self.after_step()
                    # => after epoch
                    self.after_epoch()
            # => after train
            self.after_train()
            import datetime

            self.logger.info(f"Training finished at {datetime.datetime.now()}")
            if self.cfg.enable_wandb and is_wandb_active():
                wandb.finish()

    def run_step(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            auto_cast = partial(torch.amp.autocast, device_type="cuda")
        else:
            # deprecated warning
            auto_cast = torch.cuda.amp.autocast

        input_dict = self.comm_info["input_dict"]
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)
        with auto_cast(
            enabled=self.cfg.enable_amp, dtype=AMP_DTYPE[self.cfg.amp_dtype]
        ):
            # give epoch info
            input_dict["epoch_progress"] = self.epoch / self.max_epoch
            output_dict = self.model(input_dict)
            loss = output_dict["loss"]
        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.scaler.step(self.optimizer)

            # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
            # Fix torch warning scheduler step before optimizer step.
            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            loss.backward()
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.optimizer.step()
            self.scheduler.step()
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict

    def after_epoch(self):
        if self.cfg.empty_cache_per_epoch:
            torch.cuda.empty_cache()
        for h in self.hooks:
            h.after_epoch()
            if self.cfg.empty_cache_per_epoch:
                torch.cuda.empty_cache()
        self.storage.reset_histories()

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Total params: {n_parameters}")
        # check cuda if is available
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available!")
        else:
            self.logger.info("CUDA is available!")
            self.logger.info(f"torch.cuda version: {torch.version.cuda}")

        # ADD THIS: Debug hook registration
        if self.cfg.get('nan_detection', False):  # Add a config flag to enable/disable
            self._register_nan_detection_hooks(model)

        # check if multi-gpu is available
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        if self.cfg.enable_wandb and comm.is_main_process():
            wandb_name, wandb_tag = _resolve_wandb_run_name_and_tag(self.cfg)
            wandb.init(
                project=self.cfg.wandb_project,
                name=wandb_name,
                tags=[wandb_tag],
                dir=self.cfg.save_path,
                settings=wandb.Settings(api_key=self.cfg.wandb_key, code_dir="."),
                config=self.cfg,
                resume='must' if self.cfg.wandb_id is not None else None,
                id=self.cfg.wandb_id,
            )
            wandb.run.log_code(".")
            wandb.watch(self.model)
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=True,
            persistent_workers=True,
        )
        return train_loader

    def build_val_loader(self):
        if not self.cfg.evaluate:
            return None

        val_cfg = self.cfg.data.val
        val_cfg = val_cfg if isinstance(val_cfg, (list, tuple)) else [val_cfg]

        loaders = []
        for cfg_i in val_cfg:
            val_data = build_dataset(cfg_i)
             # max_scenes limitation if specified
            if hasattr(cfg_i, 'max_scenes') and cfg_i.max_scenes is not None:
                max_scenes = min(cfg_i.max_scenes, len(val_data))
                val_data = torch.utils.data.Subset(val_data, range(max_scenes))
            sampler = (
                torch.utils.data.distributed.DistributedSampler(val_data)
                if comm.get_world_size() > 1
                else None
            )
            loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=False,
                sampler=sampler,
                collate_fn=collate_fn,
            )
            loaders.append(loader)

        return loaders[0] if len(loaders) == 1 else loaders # compatible with LangPretrainZeroShotSemSegEval

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.eval_epoch
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        scaler = torch.amp.GradScaler("cuda") if self.cfg.enable_amp else None
        return scaler
    
    def _register_nan_detection_hooks(self, model):
        """Register hooks to detect NaN/Inf values in forward pass"""
        self.first_nan_detected = False
        
        def check_nan_hook(module, input, output):
            if self.first_nan_detected:
                return  # Skip if we already found the first NaN
                
            # Check output
            if isinstance(output, torch.Tensor):
                if not torch.isfinite(output).all():
                    self.first_nan_detected = True
                    nan_count = torch.isnan(output).sum().item()
                    inf_count = torch.isinf(output).sum().item()
                    
                    self.logger.error(f"🚨 Non-finite values detected in {module.__class__.__name__}")
                    self.logger.error(f"   Module: {module}")
                    self.logger.error(f"   NaN count: {nan_count}, Inf count: {inf_count}")
                    self.logger.error(f"   Output shape: {output.shape}")
                    self.logger.error(f"   Output dtype: {output.dtype}")
                    self.logger.error(f"   Output stats: min={output[torch.isfinite(output)].min().item() if torch.isfinite(output).any() else 'N/A'}, "
                                    f"max={output[torch.isfinite(output)].max().item() if torch.isfinite(output).any() else 'N/A'}")
                    
                    # Check input too
                    if isinstance(input, tuple) and len(input) > 0 and isinstance(input[0], torch.Tensor):
                        input_tensor = input[0]
                        if not torch.isfinite(input_tensor).all():
                            self.logger.error(f"   ⚠️ Input also has non-finite values!")
                        else:
                            self.logger.error(f"   ✓ Input is finite (min={input_tensor.min().item()}, max={input_tensor.max().item()})")
                    
                    # Optional: Save checkpoint for debugging
                    if self.cfg.get('save_on_nan', False):
                        torch.save({
                            'module_state': module.state_dict(),
                            'input': input,
                            'output': output,
                        }, f"{self.cfg.save_path}/nan_debug_checkpoint.pth")
                        self.logger.error(f"   Saved debug checkpoint to {self.cfg.save_path}/nan_debug_checkpoint.pth")
                    
                    # Optional: Enter debugger
                    if self.cfg.get('breakpoint_on_nan', False):
                        import pdb; pdb.set_trace()
        
        # Register hook on all modules
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                module.register_forward_hook(check_nan_hook)
        
        self.logger.info(f"✓ Registered NaN detection hooks on {len(list(model.modules()))} modules")


@TRAINERS.register_module("MultiDatasetTrainer")
class MultiDatasetTrainer(Trainer):
    def build_train_loader(self):
        from pointcept.datasets import MultiDatasetDataloader

        train_data = build_dataset(self.cfg.data.train)
        train_loader = MultiDatasetDataloader(
            train_data,
            self.cfg.batch_size_per_gpu,
            self.cfg.num_worker_per_gpu,
            self.cfg.mix_prob,
            self.cfg.seed,
        )

        # simulate a single epoch length without materializing the data:
        main_len = len(train_loader.dataloaders[0])  # number of batches in dataset[0]
        ratio0 = train_loader.ratios[0]
        full_outer, rem = divmod(main_len, ratio0)
        true_iters = full_outer * sum(train_loader.ratios) + rem

        self.comm_info["iter_per_epoch"] = true_iters
        return train_loader

    def build_scheduler(self):
        iters_per_epoch = self.comm_info.get("iter_per_epoch", len(self.train_loader))
        total_steps = iters_per_epoch * (self.max_epoch - self.start_epoch)
        self.cfg.scheduler.total_steps = total_steps
        self.logger.info(f"Total steps for scheduler: {total_steps}")
        return build_scheduler(self.cfg.scheduler, self.optimizer)
