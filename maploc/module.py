# Copyright (c) Meta Platforms, Inc. and affiliates.

from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from torchmetrics import MeanMetric, MetricCollection

from . import logger
from .models import get_model


class AverageKeyMeter(MeanMetric):
    def __init__(self, key, *args, **kwargs):
        self.key = key
        super().__init__(*args, **kwargs)

    def update(self, dict):
        value = dict[self.key]
        value = value[torch.isfinite(value)]
        return super().update(value)


class GenericModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        name = cfg.model.get("name")
        name = "orienternet" if name in ("localizer_bev_depth", None) else name
        self.model = get_model(name)(cfg.model)
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.metrics_val = MetricCollection(self.model.metrics(), prefix="val/")
        self.losses_val = None  # we do not know the loss keys in advance

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch):
        pred = self(batch)
        losses = self.model.loss(pred, batch)
        self.log_dict(
            {f"loss/{k}/train": v.mean() for k, v in losses.items()},
            prog_bar=True,
            rank_zero_only=True,
        )
        return losses["total"].mean()

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        losses = self.model.loss(pred, batch)
        if self.losses_val is None:
            self.losses_val = MetricCollection(
                {k: AverageKeyMeter(k).to(self.device) for k in losses},
                prefix="loss/",
                postfix="/val",
            )
        self.metrics_val(pred, batch)
        self.log_dict(self.metrics_val, sync_dist=True)
        self.losses_val.update(losses)
        self.log_dict(self.losses_val, sync_dist=True)

    def validation_epoch_start(self, batch):
        self.losses_val = None

    def configure_optimizers(self):
        '''
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.cfg.training.lr, 
            weight_decay=self.cfg.training.get('weight_decay', 1e-4),
            betas=(0.9, 0.999),
            eps=1e-8
            )
        '''
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.cfg.training.lr, 
            weight_decay=self.cfg.training.get('weight_decay', 1e-5)
            )   
        ret = {"optimizer": optimizer}
        cfg_scheduler = self.cfg.training.get("lr_scheduler")
        if cfg_scheduler is not None:
            
            scheduler_class = getattr(torch.optim.lr_scheduler, cfg_scheduler.name)
            
            if cfg_scheduler.name == "CosineAnnealingLR":
                scheduler_kwargs = {
                    "T_max" : cfg_scheduler.get("T_max",10),
                    "eta_min": cfg_scheduler.get("eta_min",0)
                }

            elif cfg_scheduler.name == "StepLR":
                scheduler_kwargs = {
                    'step_size': cfg_scheduler.get('step_size', 50),
                    'gamma': cfg_scheduler.get('gamma', 0.1)
                }
            elif cfg_scheduler.name == "ExponentialLR":
                scheduler_kwargs = {
                    'gamma': cfg_scheduler.get('gamma', 0.95)
                }

            elif cfg_scheduler.name == "OneCycleLR":
                
                if hasattr(self.trainer, 'estimated_stepping_batches'):
                    total_steps = self.trainer.estimated_stepping_batches
                else:
                    max_epochs = self.cfg.training.trainer.max_epochs
                    steps_per_epoch = 22642
                    total_steps = steps_per_epoch * max_epochs
                
                scheduler_kwargs = {
                    'max_lr': cfg_scheduler.get('max_lr', self.cfg.training.lr),
                    'total_steps': total_steps,
                    'pct_start': cfg_scheduler.get('pct_start', 0.15),
                    'anneal_strategy': cfg_scheduler.get('anneal_strategy', 'cos'),
                    'div_factor': cfg_scheduler.get('div_factor', 25),
                    'final_div_factor': cfg_scheduler.get('final_div_factor', 1000),
                    'three_phase': cfg_scheduler.get('three_phase', False),
                }
                
                scheduler = scheduler_class(optimizer=optimizer, **scheduler_kwargs)
                ret["lr_scheduler"] = {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "name": "learning_rate",
                }
                return ret


            elif cfg_scheduler.name == "ReduceLROnPlateau":
                    # ReduceLROnPlateau
                    scheduler_kwargs = {
                        'mode': cfg_scheduler.get('mode', 'min'),
                        'factor': cfg_scheduler.get('factor', 0.7),
                        'patience': cfg_scheduler.get('patience', 2),
                        'threshold': cfg_scheduler.get('threshold', 0.01),
                        'threshold_mode': cfg_scheduler.get('threshold_mode', 'rel'),
                        'cooldown': cfg_scheduler.get('cooldown', 0),
                        'min_lr': cfg_scheduler.get('min_lr', 5e-7),
                        'eps': cfg_scheduler.get('eps', 1e-8),
                    }
                    
                    lr_scheduler_config = {
                        "scheduler": scheduler_class(optimizer=optimizer, **scheduler_kwargs),
                        "interval": "epoch",
                        "frequency": 1,
                        "monitor": cfg_scheduler.get("monitor", "loss/total/val"),  # 🔧 监控的指标
                        "strict": cfg_scheduler.get("strict", True),
                        "name": "learning_rate",
                    }

            elif cfg_scheduler.name == "CosineAnnealingWarmRestarts":
                # CosineAnnealingWarmRestarts
                scheduler_kwargs = {
                    'T_0': cfg_scheduler.get('T_0', 10),
                    'T_mult': cfg_scheduler.get('T_mult', 2),
                    'eta_min': cfg_scheduler.get('eta_min', 1e-6),
                }
                lr_scheduler_config = {
                    "scheduler": scheduler_class(optimizer=optimizer, **scheduler_kwargs),
                    "interval": "epoch",
                    "frequency": 1,
                    "name": "learning_rate",
                }

            else:
                excluded_keys={"name", "max_epochs", "warmup_steps"}
                scheduler_kwargs = {
                    k: v for k, v in cfg_scheduler.items() 
                    if k not in excluded_keys
                }
            
            scheduler = scheduler_class(optimizer=optimizer, **scheduler_kwargs)

            ret["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "loss/total/val",
                "strict": True,
                "name": "learning_rate",
            }
        return ret

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location=None,
        hparams_file=None,
        strict=False,
        cfg=None,
        find_best=False,
    ):
        assert hparams_file is None, "hparams are not supported."

        checkpoint = torch.load(
            checkpoint_path, map_location=map_location or (lambda storage, loc: storage), weights_only=False
        )
        if find_best:
            best_score, best_name = None, None
            modes = {"min": torch.lt, "max": torch.gt}
            for key, state in checkpoint["callbacks"].items():
                if not key.startswith("ModelCheckpoint"):
                    continue
                mode = eval(key.replace("ModelCheckpoint", ""))["mode"]
                if best_score is None or modes[mode](
                    state["best_model_score"], best_score
                ):
                    best_score = state["best_model_score"]
                    best_name = Path(state["best_model_path"]).name
            logger.info("Loading best checkpoint %s", best_name)
            if best_name != checkpoint_path:
                return cls.load_from_checkpoint(
                    Path(checkpoint_path).parent / best_name,
                    map_location,
                    hparams_file,
                    strict,
                    cfg,
                    find_best=False,
                )

        logger.info(
            "Using checkpoint %s from epoch %d and step %d.",
            checkpoint_path.name,
            checkpoint["epoch"],
            checkpoint["global_step"],
        )
        cfg_ckpt = checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY]
        if list(cfg_ckpt.keys()) == ["cfg"]:  # backward compatibility
            cfg_ckpt = cfg_ckpt["cfg"]
        cfg_ckpt = OmegaConf.create(cfg_ckpt)

        if cfg is None:
            cfg = {}
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        with open_dict(cfg_ckpt):
            cfg = OmegaConf.merge(cfg_ckpt, cfg)

        return pl.core.saving._load_state(cls, checkpoint, strict=strict, cfg=cfg)