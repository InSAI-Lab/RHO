# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.distributed as dist
import os

from omegaconf import DictConfig, OmegaConf

from .. import logger
from ..conf import data as conf_data_dir
from ..data import MapillaryPanoMixedDataModule
from .run_pano import evaluate

split_overrides = {
    "val": {
        "scenes":[
            "RHO_clean/detroit",
            "RHO_clean/berlin",
            "RHO_clean/chicago",
            "RHO_clean/washington",
            "RHO_clean/sanfrancisco",
            "RHO_clean/toulouse",
            "RHO_clean/montrouge",
            "RHO_rainy/detroit",
            "RHO_rainy/berlin",
            "RHO_rainy/chicago",
            "RHO_rainy/washington",
            "RHO_rainy/sanfrancisco",
            "RHO_rainy/toulouse",
            "RHO_rainy/montrouge",
            "RHO_night/detroit",
            "RHO_night/berlin",
            "RHO_night/chicago",
            "RHO_night/washington",
            "RHO_night/sanfrancisco",
            "RHO_night/toulouse",
            "RHO_night/montrouge",
            "RHO_foggy/detroit",
            "RHO_foggy/berlin",
            "RHO_foggy/chicago",
            "RHO_foggy/washington",
            "RHO_foggy/sanfrancisco",
            "RHO_foggy/toulouse",
            "RHO_foggy/montrouge",
            "RHO_snowy/detroit",
            "RHO_snowy/berlin",
            "RHO_snowy/chicago",
            "RHO_snowy/washington",
            "RHO_snowy/sanfrancisco",
            "RHO_snowy/toulouse",
            "RHO_snowy/montrouge",
            "RHO_over_exposure/detroit",
            "RHO_over_exposure/berlin",
            "RHO_over_exposure/chicago",
            "RHO_over_exposure/washington",
            "RHO_over_exposure/sanfrancisco",
            "RHO_over_exposure/toulouse",
            "RHO_over_exposure/montrouge",
            "RHO_under_exposure/detroit",
            "RHO_under_exposure/berlin",
            "RHO_under_exposure/chicago",
            "RHO_under_exposure/washington",
            "RHO_under_exposure/sanfrancisco",
            "RHO_under_exposure/toulouse",
            "RHO_under_exposure/montrouge",
            "RHO_motion_blur/detroit",
            "RHO_motion_blur/berlin",
            "RHO_motion_blur/chicago",
            "RHO_motion_blur/washington",
            "RHO_motion_blur/sanfrancisco",
            "RHO_motion_blur/toulouse",
            "RHO_motion_blur/montrouge",
        ]
    
    },
}
data_cfg_train = OmegaConf.load(Path(conf_data_dir.__file__).parent / "mapillary_pano_mixed.yaml")
data_cfg = OmegaConf.merge(
    data_cfg_train,
    {
        "return_gps": True,
        "add_map_mask": True,
        "max_init_error": 32,
        "loading": {"val": {"batch_size": 3, "num_workers": 0}},
    },
)
default_cfg_single = OmegaConf.create({"data": data_cfg})
default_cfg_sequential = OmegaConf.create(
    {
        **default_cfg_single,
        "chunking": {
            "max_length": 10,
        },
    }
)


def run(
    split: str,
    experiment: str,
    cfg: Optional[DictConfig] = None,
    sequential: bool = False,
    thresholds: Tuple[int] = (1, 3, 5),
    **kwargs,
):
    cfg = cfg or {}
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    default = default_cfg_sequential if sequential else default_cfg_single
    default = OmegaConf.merge(default, dict(data=split_overrides[split]))
    cfg = OmegaConf.merge(default, cfg)
    dataset = MapillaryPanoMixedDataModule(cfg.get("data", {}))

    metrics = evaluate(experiment, cfg, dataset, split, sequential=sequential, **kwargs)

    keys = [
        "xy_max_error",
        "xy_gps_error",
        "yaw_max_error",
    ]
    if sequential:
        keys += [
            "xy_seq_error",
            "xy_gps_seq_error",
            "yaw_seq_error",
            "yaw_gps_seq_error",
        ]
    for k in keys:
        if k not in metrics:
            logger.warning("Key %s not in metrics.", k)
            continue
        rec = metrics[k].recall(thresholds).double().numpy().round(2).tolist()
        logger.info("Recall %s: %s at %s m/°", k, rec, thresholds)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val"])
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--num", type=int)
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_args()
    cfg = OmegaConf.from_cli(args.dotlist)
    run(
        args.split,
        args.experiment,
        cfg,
        args.sequential,
        output_dir=args.output_dir,
        num=args.num,
    )