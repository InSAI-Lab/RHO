# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
from pathlib import Path
from typing import Optional, Tuple

from omegaconf import DictConfig, OmegaConf

from .. import logger
from ..conf import data as conf_data_dir
from ..data import MapillaryPanoMixedDataModule
from .run_pano import evaluate

split_overrides = {
    "val": {
        "scenes":[
            "MGL_rainy/detroit",
            "MGL_rainy/berlin",
            "MGL_rainy/chicago",
            "MGL_rainy/washington",
            "MGL_rainy/sanfrancisco",
            "MGL_rainy/toulouse",
            "MGL_rainy/montrouge",
            "MGL_night/detroit",
            "MGL_night/berlin",
            "MGL_night/chicago",
            "MGL_night/washington",
            "MGL_night/sanfrancisco",
            "MGL_night/toulouse",
            "MGL_night/montrouge",
            "MGL_foggy/detroit",
            "MGL_foggy/berlin",
            "MGL_foggy/chicago",
            "MGL_foggy/washington",
            "MGL_foggy/sanfrancisco",
            "MGL_foggy/toulouse",
            "MGL_foggy/montrouge",
            "MGL_snowy/detroit",
            "MGL_snowy/berlin",
            "MGL_snowy/chicago",
            "MGL_snowy/washington",
            "MGL_snowy/sanfrancisco",
            "MGL_snowy/toulouse",
            "MGL_snowy/montrouge",
            "MGL_over_exposure/detroit",
            "MGL_over_exposure/berlin",
            "MGL_over_exposure/chicago",
            "MGL_over_exposure/washington",
            "MGL_over_exposure/sanfrancisco",
            "MGL_over_exposure/toulouse",
            "MGL_over_exposure/montrouge",
            "MGL_under_exposure/detroit",
            "MGL_under_exposure/berlin",
            "MGL_under_exposure/chicago",
            "MGL_under_exposure/washington",
            "MGL_under_exposure/sanfrancisco",
            "MGL_under_exposure/toulouse",
            "MGL_under_exposure/montrouge",
            "MGL_motion_blur/detroit",
            "MGL_motion_blur/berlin",
            "MGL_motion_blur/chicago",
            "MGL_motion_blur/washington",
            "MGL_motion_blur/sanfrancisco",
            "MGL_motion_blur/toulouse",
            "MGL_motion_blur/montrouge",
        ]
    
    },
}
data_cfg_train = OmegaConf.load(Path(conf_data_dir.__file__).parent / "mapillary_pano_all_noise.yaml")
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