import sys
from pathlib import Path
import os
import torch
import yaml
from torchmetrics import MetricCollection
from omegaconf import OmegaConf as OC
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pytorch_lightning import seed_everything

import maploc
from maploc.data import MapillaryDataModule
from maploc.data.torch import unbatch_to_device
from maploc.module import GenericModule
from maploc.models.metrics import Location2DError, AngleError
from maploc.evaluation.run import resolve_checkpoint_path
from maploc.evaluation.viz import plot_example_single

from maploc.models.voting import argmax_xyr, fuse_gps
from maploc.osm.viz import Colormap, plot_nodes
from maploc.utils.viz_2d import plot_images, features_to_RGB, save_plot, add_text
from maploc.utils.viz_localization import likelihood_overlay, plot_pose, plot_dense_rotations, add_circle_inset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
torch.set_grad_enabled(False)
plt.rcParams.update({'figure.max_open_warning': 0})

conf = OC.load(Path(maploc.__file__).parent / 'conf/data/mapillary_washington_pano.yaml')
conf = OC.merge(conf, OC.create(yaml.full_load("""
data_dir: "datasets/RHO_clean"
loading:
    val: {batch_size: 1, num_workers: 0}
    train: ${.val}
add_map_mask: true
return_gps: true
""")))
OC.resolve(conf)
dataset = MapillaryDataModule(conf)
dataset.prepare_data()
dataset.setup()
sampler = None

# experiment = "experiment_name"  # find the best checkpoint
# experiment = "experiment_name/checkpoint-step=N.ckpt"  # a given checkpoint
path = resolve_checkpoint_path(experiment)
print(path)
cfg = {'model': {"x_max": 55, "num_rotations": 360, "apply_map_prior": True}}
model = GenericModule.load_from_checkpoint(
    path, strict=True, find_best=not experiment.endswith('.ckpt'), cfg=cfg)
model = model.eval().cuda()
assert model.cfg.data.resize_image == dataset.cfg.resize_image

out_dir = Path('viz_kitti/washinton_360')
if out_dir is not None:
    os.makedirs(out_dir, exist_ok=True)

seed_everything(25) # best = 25
loader = dataset.dataloader("val", shuffle=sampler is None, sampler=sampler)
metrics = MetricCollection(model.model.metrics()).to(model.device)
metrics["xy_gps_error"] = Location2DError("uv_gps", model.cfg.model.pixel_per_meter)
for i, batch in zip(range(5), loader):
    pred = data = batch_ = None    
    batch_ = model.transfer_batch_to_device(batch, model.device, i)
    pred = model(batch_)
    pred = {k:v.float() if isinstance(v, torch.HalfTensor) else v for k,v in pred.items()}
    pred["uv_gps"] = batch["uv_gps"]
    loss = model.model.loss(pred, batch_)
    results = metrics(pred, batch_)
    results.pop("xy_expectation_error")
    for k in list(results):
        if "recall" in k:
            results.pop(k)
    print(f'{i} {loss["total"].item():.2f}', {k: round(v.item(), 2) for k, v in results.items()})
#     if results["xy_max_error"] < 5:
#         continue

    pred = unbatch_to_device(pred)
    data = unbatch_to_device(batch)
    plot_example_single(i, model, pred, data, results, plot_bev=True, out_dir=out_dir, show_gps=True)
    
    continue
    scales_scores = pred['pixel_scales']
    log_prob = torch.nn.functional.log_softmax(scales_scores, dim=-1)
    scales_exp = torch.sum(log_prob.exp() * torch.arange(scales_scores.shape[-1]), -1)
    total_score = torch.logsumexp(scales_scores, -1)
    plot_images([log_prob.max(-1).values.exp(), scales_exp, total_score], cmaps='jet')
    plt.show()