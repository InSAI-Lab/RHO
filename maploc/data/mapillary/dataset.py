# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import os
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as torchdata
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from ... import DATASETS_PATH, logger
from ...osm.tiling import TileManager
from ..dataset import MapLocDataset
from ..sequential import chunk_sequence
from ..torch import collate, worker_init_fn, DistributedGroupSampler


def pack_dump_dict(dump):
    for per_seq in dump.values():
        if "points" in per_seq:
            for chunk in list(per_seq["points"]):
                points = per_seq["points"].pop(chunk)
                if points is not None:
                    per_seq["points"][chunk] = np.array(
                        per_seq["points"][chunk], np.float64
                    )
        for view in per_seq["views"].values():
            for k in ["R_c2w", "roll_pitch_yaw"]:
                view[k] = np.array(view[k], np.float32)
            for k in ["chunk_id"]:
                if k in view:
                    view.pop(k)
        if "observations" in view:
            view["observations"] = np.array(view["observations"])
        for camera in per_seq["cameras"].values():
            for k in ["params"]:
                camera[k] = np.array(camera[k], np.float32)
    return dump

class Sim2RealDataModule(pl.LightningDataModule):
    dump_filename = "dump.json"
    # images_archive = "images.tar.gz"
    images_dirname = "images/"

    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "sim2real",
        # paths and fetch
        "data_dir": DATASETS_PATH / "Sim2Real",
        "local_dir": None,
        "tiles_filename": "tiles.pkl",
        "scenes": "???",
        "split": None,
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 1, "num_workers": 0},
        },
        "filter_for": None,
        "filter_by_ground_angle": None,
        "min_num_points": "???",
    }

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.local_dir = self.cfg.local_dir or os.environ.get("TMPDIR")
        if self.local_dir is not None:
            self.local_dir = Path(self.local_dir, "Sim2Real")
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")

    def prepare_data(self):
        for scene in self.cfg.scenes:
            dump_dir = self.root / scene
            assert (dump_dir / self.dump_filename).exists(), dump_dir
            assert (dump_dir / self.cfg.tiles_filename).exists(), dump_dir
            if self.local_dir is None:
                assert (dump_dir / self.images_dirname).exists(), dump_dir
                continue
            # Cache the folder of images locally to speed up reading
            local_dir = self.local_dir / scene
            if local_dir.exists():
                shutil.rmtree(local_dir)
            local_dir.mkdir(exist_ok=True, parents=True)
            # images_archive = dump_dir / self.images_archive
            # logger.info("Extracting the image archive %s.", images_archive)
            # with tarfile.open(images_archive) as fp:
                # fp.extractall(local_dir)

    def setup(self, stage: Optional[str] = None):
        self.dumps = {}
        self.tile_managers = {}
        self.image_dirs = {}
        names = []

        for scene in self.cfg.scenes:
            logger.info("Loading scene %s.", scene)
            dump_dir = self.root / scene
            logger.info("Loading map tiles %s.", self.cfg.tiles_filename)
            self.tile_managers[scene] = TileManager.load(
                dump_dir / self.cfg.tiles_filename
            )
            groups = self.tile_managers[scene].groups
            if self.cfg.num_classes:  # check consistency
                if set(groups.keys()) != set(self.cfg.num_classes.keys()):
                    raise ValueError(
                        "Inconsistent groups: "
                        f"{groups.keys()} {self.cfg.num_classes.keys()}"
                    )
                for k in groups:
                    if len(groups[k]) != self.cfg.num_classes[k]:
                        raise ValueError(
                            f"{k}: {len(groups[k])} vs {self.cfg.num_classes[k]}"
                        )
            ppm = self.tile_managers[scene].ppm
            if ppm != self.cfg.pixel_per_meter:
                raise ValueError(
                    "The tile manager and the config/model have different ground "
                    f"resolutions: {ppm} vs {self.cfg.pixel_per_meter}"
                )

            logger.info("Loading dump json file %s.", self.dump_filename)
            with (dump_dir / self.dump_filename).open("r") as fp:
                self.dumps[scene] = pack_dump_dict(json.load(fp))
            for seq, per_seq in self.dumps[scene].items():
                for cam_id, cam_dict in per_seq["cameras"].items():
                    if cam_dict["model"] != "PINHOLE":
                        raise ValueError(
                            "Unsupported camera model: "
                            f"{cam_dict['model']} for {scene},{seq},{cam_id}"
                        )

            self.image_dirs[scene] = (
                # (self.local_dir or self.root) / scene / self.images_dirname
                self.root / scene / self.images_dirname
            )
            assert self.image_dirs[scene].exists(), self.image_dirs[scene]

            for seq, data in self.dumps[scene].items():
                for name in data["views"]:
                    names.append((scene, seq, name))

        self.parse_splits(self.cfg.split, names)
        if self.cfg.filter_for is not None:
            self.filter_elements()
        self.pack_data()

    def pack_data(self):
        # We pack the data into compact tensors
        # that can be shared across processes without copy.
        exclude = {
            "compass_angle",
            "compass_accuracy",
            "gps_accuracy",
            "chunk_key",
            "panorama_offset",
        }
        cameras = {
            scene: {seq: per_seq["cameras"] for seq, per_seq in per_scene.items()}
            for scene, per_scene in self.dumps.items()
        }
        points = {
            scene: {
                seq: {
                    i: torch.from_numpy(p) for i, p in per_seq.get("points", {}).items()
                }
                for seq, per_seq in per_scene.items()
            }
            for scene, per_scene in self.dumps.items()
        }
        self.data = {}
        for stage, names in self.splits.items():
            view = self.dumps[names[0][0]][names[0][1]]["views"][names[0][2]]
            data = {k: [] for k in view.keys() - exclude}
            for scene, seq, name in names:
                for k in data:
                    data[k].append(self.dumps[scene][seq]["views"][name].get(k, None))
            for k in data:
                v = np.array(data[k])
                if np.issubdtype(v.dtype, np.integer) or np.issubdtype(
                    v.dtype, np.floating
                ):
                    v = torch.from_numpy(v)
                data[k] = v
            data["cameras"] = cameras
            data["points"] = points
            self.data[stage] = data
            self.splits[stage] = np.array(names)

    def filter_elements(self):
        for stage, names in self.splits.items():
            names_select = []
            for scene, seq, name in names:
                view = self.dumps[scene][seq]["views"][name]
                if self.cfg.filter_for == "ground_plane":
                    if not (1.0 <= view["height"] <= 3.0):
                        continue
                    planes = self.dumps[scene][seq].get("plane")
                    if planes is not None:
                        inliers = planes[str(view["chunk_id"])][-1]
                        if inliers < 10:
                            continue
                    if self.cfg.filter_by_ground_angle is not None:
                        plane = np.array(view["plane_params"])
                        normal = plane[:3] / np.linalg.norm(plane[:3])
                        angle = np.rad2deg(np.arccos(np.abs(normal[-1])))
                        if angle > self.cfg.filter_by_ground_angle:
                            continue
                elif self.cfg.filter_for == "pointcloud":
                    if len(view["observations"]) < self.cfg.min_num_points:
                        continue
                elif self.cfg.filter_for is not None:
                    raise ValueError(f"Unknown filtering: {self.cfg.filter_for}")
                names_select.append((scene, seq, name))
            logger.info(
                "%s: Keep %d/%d images after filtering for %s.",
                stage,
                len(names_select),
                len(names),
                self.cfg.filter_for,
            )
            self.splits[stage] = names_select

    def parse_splits(self, split_arg, names):
        if split_arg is None:
            self.splits = {
                "train": names,
                "val": names,
            }
        elif isinstance(split_arg, int):
            names = np.random.RandomState(self.cfg.seed).permutation(names).tolist()
            self.splits = {
                "train": names[split_arg:],
                "val": names[:split_arg],
            }
        elif isinstance(split_arg, DictConfig):
            scenes_val = set(split_arg.val)
            scenes_train = set(split_arg.train)
            assert len(scenes_val - set(self.cfg.scenes)) == 0
            assert len(scenes_train - set(self.cfg.scenes)) == 0
            self.splits = {
                "train": [n for n in names if n[0] in scenes_train],
                "val": [n for n in names if n[0] in scenes_val],
            }
        elif isinstance(split_arg, str):
            if (self.root / split_arg).exists():
                # Common split file.
                with (self.root / split_arg).open("r") as fp:
                    splits = json.load(fp)
            else:
                # Per-scene split file.
                splits = defaultdict(dict)
                for scene in self.cfg.scenes:
                    with (self.root / split_arg.format(scene=scene)).open("r") as fp:
                        scene_splits = json.load(fp)
                    for split_name in scene_splits:
                        splits[split_name][scene] = scene_splits[split_name]
            splits = {
                split_name: {scene: set(ids) for scene, ids in split.items()}
                for split_name, split in splits.items()
            }
            self.splits = {}
            for split_name, split in splits.items():
                self.splits[split_name] = [
                    (scene, *arg, name)
                    for scene, *arg, name in names
                    if scene in split and int(name.rsplit("_", 1)[0]) in split[scene]
                ]
        else:
            raise ValueError(split_arg)

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.splits[stage],
            self.data[stage],
            self.image_dirs,
            self.tile_managers,
            image_ext=".jpg",
        )

    def dataloader(
        self,
        stage: str,
        shuffle: bool = False,
        num_workers: int = None,
        sampler: Optional[torchdata.Sampler] = None,
    ):
        dataset = self.dataset(stage)
        cfg = self.cfg["loading"][stage]
        num_workers = cfg["num_workers"] if num_workers is None else num_workers
        loader = torchdata.DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=num_workers,
            shuffle=shuffle or (stage == "train"),
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)

    def sequence_dataset(self, stage: str, **kwargs):
        keys = self.splits[stage]
        seq2indices = defaultdict(list)
        for index, (_, seq, _) in enumerate(keys):
            seq2indices[seq].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(self.data[stage], indices, **kwargs)
            for i, sub_indices in enumerate(chunks):
                chunk2indices[seq, i] = sub_indices
        # store the index of each chunk in its sequence
        chunk_indices = torch.full((len(keys),), -1)
        for (_, chunk_index), idx in chunk2indices.items():
            chunk_indices[idx] = chunk_index
        self.data[stage]["chunk_index"] = chunk_indices
        dataset = self.dataset(stage)
        return dataset, chunk2indices

    def sequence_dataloader(self, stage: str, shuffle: bool = False, **kwargs):
        dataset, chunk2idx = self.sequence_dataset(stage, **kwargs)
        chunk_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(chunk_keys))
            chunk_keys = [chunk_keys[i] for i in perm]
        key_indices = [i for key in chunk_keys for i in chunk2idx[key]]
        num_workers = self.cfg["loading"][stage]["num_workers"]
        loader = torchdata.DataLoader(
            dataset,
            batch_size=None,
            sampler=key_indices,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return loader, chunk_keys, chunk2idx

class Mapillary2KValDataModule(pl.LightningDataModule):
    dump_filename = "dump.json"
    # images_archive = "images.tar.gz"
    images_dirname = "images/"

    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "mapillary_2kval",
        # paths and fetch
        "data_dir": DATASETS_PATH / "MGL",
        "local_dir": None,
        "tiles_filename": "tiles.pkl",
        "scenes": "???",
        "split": None,
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 1, "num_workers": 0},
        },
        "filter_for": None,
        "filter_by_ground_angle": None,
        "min_num_points": "???",
    }

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.local_dir = self.cfg.local_dir or os.environ.get("TMPDIR")
        if self.local_dir is not None:
            self.local_dir = Path(self.local_dir, "MGL")
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")

    def prepare_data(self):
        for scene in self.cfg.scenes:
            dump_dir = self.root / scene
            assert (dump_dir / self.dump_filename).exists(), dump_dir
            assert (dump_dir / self.cfg.tiles_filename).exists(), dump_dir
            if self.local_dir is None:
                assert (dump_dir / self.images_dirname).exists(), dump_dir
                continue
            # Cache the folder of images locally to speed up reading
            local_dir = self.local_dir / scene
            if local_dir.exists():
                shutil.rmtree(local_dir)
            local_dir.mkdir(exist_ok=True, parents=True)
            # images_archive = dump_dir / self.images_archive
            # logger.info("Extracting the image archive %s.", images_archive)
            # with tarfile.open(images_archive) as fp:
                # fp.extractall(local_dir)

    def setup(self, stage: Optional[str] = None):
        self.dumps = {}
        self.tile_managers = {}
        self.image_dirs = {}
        names = []

        for scene in self.cfg.scenes:
            logger.info("Loading scene %s.", scene)
            dump_dir = self.root / scene
            logger.info("Loading map tiles %s.", self.cfg.tiles_filename)
            self.tile_managers[scene] = TileManager.load(
                dump_dir / self.cfg.tiles_filename
            )
            groups = self.tile_managers[scene].groups
            if self.cfg.num_classes:  # check consistency
                if set(groups.keys()) != set(self.cfg.num_classes.keys()):
                    raise ValueError(
                        "Inconsistent groups: "
                        f"{groups.keys()} {self.cfg.num_classes.keys()}"
                    )
                for k in groups:
                    if len(groups[k]) != self.cfg.num_classes[k]:
                        raise ValueError(
                            f"{k}: {len(groups[k])} vs {self.cfg.num_classes[k]}"
                        )
            ppm = self.tile_managers[scene].ppm
            if ppm != self.cfg.pixel_per_meter:
                raise ValueError(
                    "The tile manager and the config/model have different ground "
                    f"resolutions: {ppm} vs {self.cfg.pixel_per_meter}"
                )

            logger.info("Loading dump json file %s.", self.dump_filename)
            with (dump_dir / self.dump_filename).open("r") as fp:
                self.dumps[scene] = pack_dump_dict(json.load(fp))
            for seq, per_seq in self.dumps[scene].items():
                for cam_id, cam_dict in per_seq["cameras"].items():
                    if cam_dict["model"] != "PINHOLE":
                        raise ValueError(
                            "Unsupported camera model: "
                            f"{cam_dict['model']} for {scene},{seq},{cam_id}"
                        )

            self.image_dirs[scene] = (
                # (self.local_dir or self.root) / scene / self.images_dirname
                self.root / scene / self.images_dirname
            )
            assert self.image_dirs[scene].exists(), self.image_dirs[scene]

            for seq, data in self.dumps[scene].items():
                for name in data["views"]:
                    names.append((scene, seq, name))

        self.parse_splits(self.cfg.split, names)
        if self.cfg.filter_for is not None:
            self.filter_elements()
        self.pack_data()

    def pack_data(self):
        # We pack the data into compact tensors
        # that can be shared across processes without copy.
        exclude = {
            "compass_angle",
            "compass_accuracy",
            "gps_accuracy",
            "chunk_key",
            "panorama_offset",
        }
        cameras = {
            scene: {seq: per_seq["cameras"] for seq, per_seq in per_scene.items()}
            for scene, per_scene in self.dumps.items()
        }
        points = {
            scene: {
                seq: {
                    i: torch.from_numpy(p) for i, p in per_seq.get("points", {}).items()
                }
                for seq, per_seq in per_scene.items()
            }
            for scene, per_scene in self.dumps.items()
        }
        self.data = {}
        for stage, names in self.splits.items():
            view = self.dumps[names[0][0]][names[0][1]]["views"][names[0][2]]
            data = {k: [] for k in view.keys() - exclude}
            for scene, seq, name in names:
                for k in data:
                    data[k].append(self.dumps[scene][seq]["views"][name].get(k, None))
            for k in data:
                v = np.array(data[k])
                if np.issubdtype(v.dtype, np.integer) or np.issubdtype(
                    v.dtype, np.floating
                ):
                    v = torch.from_numpy(v)
                data[k] = v
            data["cameras"] = cameras
            data["points"] = points
            self.data[stage] = data
            self.splits[stage] = np.array(names)

    def filter_elements(self):
        for stage, names in self.splits.items():
            names_select = []
            for scene, seq, name in names:
                view = self.dumps[scene][seq]["views"][name]
                if self.cfg.filter_for == "ground_plane":
                    if not (1.0 <= view["height"] <= 3.0):
                        continue
                    planes = self.dumps[scene][seq].get("plane")
                    if planes is not None:
                        inliers = planes[str(view["chunk_id"])][-1]
                        if inliers < 10:
                            continue
                    if self.cfg.filter_by_ground_angle is not None:
                        plane = np.array(view["plane_params"])
                        normal = plane[:3] / np.linalg.norm(plane[:3])
                        angle = np.rad2deg(np.arccos(np.abs(normal[-1])))
                        if angle > self.cfg.filter_by_ground_angle:
                            continue
                elif self.cfg.filter_for == "pointcloud":
                    if len(view["observations"]) < self.cfg.min_num_points:
                        continue
                elif self.cfg.filter_for is not None:
                    raise ValueError(f"Unknown filtering: {self.cfg.filter_for}")
                names_select.append((scene, seq, name))
            logger.info(
                "%s: Keep %d/%d images after filtering for %s.",
                stage,
                len(names_select),
                len(names),
                self.cfg.filter_for,
            )
            self.splits[stage] = names_select

    def parse_splits(self, split_arg, names):
        if split_arg is None:
            self.splits = {
                "train": names,
                "val": names,
            }
        elif isinstance(split_arg, int):
            names = np.random.RandomState(self.cfg.seed).permutation(names).tolist()
            self.splits = {
                "train": names[split_arg:],
                "val": names[:split_arg],
            }
        elif isinstance(split_arg, DictConfig):
            scenes_val = set(split_arg.val)
            scenes_train = set(split_arg.train)
            assert len(scenes_val - set(self.cfg.scenes)) == 0
            assert len(scenes_train - set(self.cfg.scenes)) == 0
            self.splits = {
                "train": [n for n in names if n[0] in scenes_train],
                "val": [n for n in names if n[0] in scenes_val],
            }
        elif isinstance(split_arg, str):
            if (self.root / split_arg).exists():
                # Common split file.
                with (self.root / split_arg).open("r") as fp:
                    splits = json.load(fp)
            else:
                # Per-scene split file.
                splits = defaultdict(dict)
                for scene in self.cfg.scenes:
                    with (self.root / split_arg.format(scene=scene)).open("r") as fp:
                        scene_splits = json.load(fp)
                    for split_name in scene_splits:
                        splits[split_name][scene] = scene_splits[split_name]
            splits = {
                split_name: {scene: set(ids) for scene, ids in split.items()}
                for split_name, split in splits.items()
            }
            self.splits = {}
            for split_name, split in splits.items():
                self.splits[split_name] = [
                    (scene, *arg, name)
                    for scene, *arg, name in names
                    if scene in split and int(name.rsplit("_", 1)[0]) in split[scene]
                ]
        else:
            raise ValueError(split_arg)

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.splits[stage],
            self.data[stage],
            self.image_dirs,
            self.tile_managers,
            image_ext=".jpg",
        )

    def dataloader(
    self,
    stage: str,
    shuffle: bool = False,
    num_workers: int = None,
    sampler: Optional[torchdata.Sampler] = None,
    ):
        dataset = self.dataset(stage)
        cfg = self.cfg["loading"][stage]
        num_workers = cfg["num_workers"] if num_workers is None else num_workers
        
        
        if stage == "val" and sampler is None:  
            total_samples = len(dataset)
            
            if total_samples > 2000:
                
                from torch.utils.data import SubsetRandomSampler
                
                generator = torch.Generator()
                generator.manual_seed(42)
                indices = torch.randperm(total_samples, generator=generator)[:2000]
                sampler = SubsetRandomSampler(indices)
                
                
                shuffle = False
                
                print(f"Validation set: Using {len(indices)} samples out of {total_samples} total samples")
        
        loader = torchdata.DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=num_workers,
            shuffle=shuffle or (stage == "train") if sampler is None else False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)

    def sequence_dataset(self, stage: str, **kwargs):
        keys = self.splits[stage]
        seq2indices = defaultdict(list)
        for index, (_, seq, _) in enumerate(keys):
            seq2indices[seq].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(self.data[stage], indices, **kwargs)
            for i, sub_indices in enumerate(chunks):
                chunk2indices[seq, i] = sub_indices
        # store the index of each chunk in its sequence
        chunk_indices = torch.full((len(keys),), -1)
        for (_, chunk_index), idx in chunk2indices.items():
            chunk_indices[idx] = chunk_index
        self.data[stage]["chunk_index"] = chunk_indices
        dataset = self.dataset(stage)
        return dataset, chunk2indices

    def sequence_dataloader(self, stage: str, shuffle: bool = False, **kwargs):
        dataset, chunk2idx = self.sequence_dataset(stage, **kwargs)
        chunk_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(chunk_keys))
            chunk_keys = [chunk_keys[i] for i in perm]
        key_indices = [i for key in chunk_keys for i in chunk2idx[key]]
        num_workers = self.cfg["loading"][stage]["num_workers"]
        loader = torchdata.DataLoader(
            dataset,
            batch_size=None,
            sampler=key_indices,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return loader, chunk_keys, chunk2idx

class Sim2RealPanoDataModule(pl.LightningDataModule):
    dump_filename = "dump.json"
    # images_archive = "images.tar.gz"
    images_dirname = "images/"

    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "sim2real",
        # paths and fetch
        "data_dir": DATASETS_PATH / "Sim2Real",
        "local_dir": None,
        "tiles_filename": "tiles.pkl",
        "scenes": "???",
        "split": None,
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 3, "num_workers": 3},
        },
        "filter_for": None,
        "filter_by_ground_angle": None,
        "min_num_points": "???",
    }

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.num_views = 3
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.local_dir = self.cfg.local_dir or os.environ.get("TMPDIR")
        if self.local_dir is not None:
            self.local_dir = Path(self.local_dir, "Sim2Real")
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")

    def prepare_data(self):
        for scene in self.cfg.scenes:
            dump_dir = self.root / scene
            assert (dump_dir / self.dump_filename).exists(), dump_dir
            assert (dump_dir / self.cfg.tiles_filename).exists(), dump_dir
            if self.local_dir is None:
                assert (dump_dir / self.images_dirname).exists(), dump_dir
                continue
            # Cache the folder of images locally to speed up reading
            local_dir = self.local_dir / scene
            if local_dir.exists():
                shutil.rmtree(local_dir)
            local_dir.mkdir(exist_ok=True, parents=True)
            # images_archive = dump_dir / self.images_archive
            # logger.info("Extracting the image archive %s.", images_archive)
            # with tarfile.open(images_archive) as fp:
                # fp.extractall(local_dir)

    def setup(self, stage: Optional[str] = None):
        self.dumps = {}
        self.tile_managers = {}
        self.image_dirs = {}
        names = []

        for scene in self.cfg.scenes:
            logger.info("Loading scene %s.", scene)
            dump_dir = self.root / scene
            logger.info("Loading map tiles %s.", self.cfg.tiles_filename)
            self.tile_managers[scene] = TileManager.load(
                dump_dir / self.cfg.tiles_filename
            )
            groups = self.tile_managers[scene].groups
            if self.cfg.num_classes:  # check consistency
                if set(groups.keys()) != set(self.cfg.num_classes.keys()):
                    raise ValueError(
                        "Inconsistent groups: "
                        f"{groups.keys()} {self.cfg.num_classes.keys()}"
                    )
                for k in groups:
                    if len(groups[k]) != self.cfg.num_classes[k]:
                        raise ValueError(
                            f"{k}: {len(groups[k])} vs {self.cfg.num_classes[k]}"
                        )
            ppm = self.tile_managers[scene].ppm
            if ppm != self.cfg.pixel_per_meter:
                raise ValueError(
                    "The tile manager and the config/model have different ground "
                    f"resolutions: {ppm} vs {self.cfg.pixel_per_meter}"
                )

            logger.info("Loading dump json file %s.", self.dump_filename)
            with (dump_dir / self.dump_filename).open("r") as fp:
                self.dumps[scene] = pack_dump_dict(json.load(fp))
            for seq, per_seq in self.dumps[scene].items():
                for cam_id, cam_dict in per_seq["cameras"].items():
                    if cam_dict["model"] != "PINHOLE":
                        raise ValueError(
                            "Unsupported camera model: "
                            f"{cam_dict['model']} for {scene},{seq},{cam_id}"
                        )

            self.image_dirs[scene] = (
                # (self.local_dir or self.root) / scene / self.images_dirname
                self.root / scene / self.images_dirname
            )
            assert self.image_dirs[scene].exists(), self.image_dirs[scene]

            for seq, data in self.dumps[scene].items():
                for name in data["views"]:
                    names.append((scene, seq, name))

        self.parse_splits(self.cfg.split, names)
        if self.cfg.filter_for is not None:
            self.filter_elements()
        self.pack_data()

    def pack_data(self):
        # We pack the data into compact tensors
        # that can be shared across processes without copy.
        # We group views by panorama to enable panorama-based batching.
        exclude = {
            "compass_angle",
            "compass_accuracy",
            "gps_accuracy",
            "chunk_key",
            "panorama_offset",
        }
        cameras = {
            scene: {seq: per_seq["cameras"] for seq, per_seq in per_scene.items()}
            for scene, per_scene in self.dumps.items()
        }
        points = {
            scene: {
                seq: {
                    i: torch.from_numpy(p) for i, p in per_seq.get("points", {}).items()
                }
                for seq, per_seq in per_scene.items()
            }
            for scene, per_scene in self.dumps.items()
        }
        self.data = {}
        self.panorama_groups = {}  # Store panorama grouping info

        for stage, names in self.splits.items():
            # Group views by panorama ID
            panorama_views = self._group_views_by_panorama(names)

            # Validate that we have complete panoramas (n views each)
            incomplete_panoramas = [
                p_id for p_id, views in panorama_views.items() if len(views) != self.num_views
            ]
            if incomplete_panoramas:
                logger.warning(
                    f"Found {len(incomplete_panoramas)} incomplete panoramas "
                    f"in {stage} set. These will be excluded. "
                    f"First few: {incomplete_panoramas[:5]}"
                )
                # Keep only complete panoramas
                panorama_views = {
                    p_id: views
                    for p_id, views in panorama_views.items()
                    if len(views) == self.num_views
                }

            # Store panorama groups for this stage
            self.panorama_groups[stage] = panorama_views

            # Create flattened names list maintaining panorama view order
            ordered_names = []
            for p_id in sorted(panorama_views.keys()):
                views = panorama_views[p_id]
                # Sort views by suffix for consistent order:
                # suffix_order = {"front": 0, "left": 1, "back": 2, "right": 3}
                # suffix_order = {"view1": 0, "view2": 1, "view3": 2}
                suffix_order = {"view1": 0, "view2": 1, "view3": 2}
                # suffix_order = {"v1": 0, "v2": 1, "v3": 2, "v4": 3, "v5": 4, "v6": 5, "v7": 6, "v8": 7, "v9": 8}

                views_sorted = sorted(
                    views, key=lambda x: suffix_order.get(x[2].split("_")[-1], 999)
                )
                ordered_names.extend(views_sorted)

            # Pack data as before but with grouped panorama ordering
            if ordered_names:
                view = self.dumps[ordered_names[0][0]][ordered_names[0][1]]["views"][
                    ordered_names[0][2]
                ]
                data = {k: [] for k in view.keys() - exclude}
                for scene, seq, name in ordered_names:
                    for k in data:
                        data[k].append(
                            self.dumps[scene][seq]["views"][name].get(k, None)
                        )
                for k in data:
                    v = np.array(data[k])
                    if np.issubdtype(v.dtype, np.integer) or np.issubdtype(
                        v.dtype, np.floating
                    ):
                        v = torch.from_numpy(v)
                    data[k] = v
                data["cameras"] = cameras
                data["points"] = points
                self.data[stage] = data
                self.splits[stage] = np.array(ordered_names)
            else:
                # Empty stage
                self.data[stage] = {"cameras": cameras, "points": points}
                self.splits[stage] = np.array([])

    def _group_views_by_panorama(self, names):
        """Group view names by panorama ID.

        Args:
            names: List of (scene, seq, name) tuples

        Returns:
            Dict mapping panorama_id to list of (scene, seq, name) tuples
        """
        panorama_views = defaultdict(list)
        for scene, seq, name in names:
            # Extract panorama ID from view name (everything before the last underscore)
            if "_" in name:
                panorama_id = name.rsplit("_", 1)[0]
                panorama_views[panorama_id].append((scene, seq, name))
            else:
                # Handle non-panoramic views if any exist
                panorama_views[name].append((scene, seq, name))
        return dict(panorama_views)

    def filter_elements(self):
        for stage, names in self.splits.items():
            names_select = []
            for scene, seq, name in names:
                view = self.dumps[scene][seq]["views"][name]
                if self.cfg.filter_for == "ground_plane":
                    if not (1.0 <= view["height"] <= 3.0):
                        continue
                    planes = self.dumps[scene][seq].get("plane")
                    if planes is not None:
                        inliers = planes[str(view["chunk_id"])][-1]
                        if inliers < 10:
                            continue
                    if self.cfg.filter_by_ground_angle is not None:
                        plane = np.array(view["plane_params"])
                        normal = plane[:3] / np.linalg.norm(plane[:3])
                        angle = np.rad2deg(np.arccos(np.abs(normal[-1])))
                        if angle > self.cfg.filter_by_ground_angle:
                            continue
                elif self.cfg.filter_for == "pointcloud":
                    if len(view["observations"]) < self.cfg.min_num_points:
                        continue
                elif self.cfg.filter_for is not None:
                    raise ValueError(f"Unknown filtering: {self.cfg.filter_for}")
                names_select.append((scene, seq, name))
            logger.info(
                "%s: Keep %d/%d images after filtering for %s.",
                stage,
                len(names_select),
                len(names),
                self.cfg.filter_for,
            )
            self.splits[stage] = names_select

    def parse_splits(self, split_arg, names):
        if split_arg is None:
            self.splits = {
                "train": names,
                "val": names,
            }
        elif isinstance(split_arg, int):
            names = np.random.RandomState(self.cfg.seed).permutation(names).tolist()
            self.splits = {
                "train": names[split_arg:],
                "val": names[:split_arg],
            }
        elif isinstance(split_arg, DictConfig):
            scenes_val = set(split_arg.val)
            scenes_train = set(split_arg.train)
            assert len(scenes_val - set(self.cfg.scenes)) == 0
            assert len(scenes_train - set(self.cfg.scenes)) == 0
            self.splits = {
                "train": [n for n in names if n[0] in scenes_train],
                "val": [n for n in names if n[0] in scenes_val],
            }
        elif isinstance(split_arg, str):
            if (self.root / split_arg).exists():
                # Common split file.
                with (self.root / split_arg).open("r") as fp:
                    splits = json.load(fp)
            else:
                # Per-scene split file.
                splits = defaultdict(dict)
                for scene in self.cfg.scenes:
                    with (self.root / split_arg.format(scene=scene)).open("r") as fp:
                        scene_splits = json.load(fp)
                    for split_name in scene_splits:
                        splits[split_name][scene] = scene_splits[split_name]
            splits = {
                split_name: {scene: set(ids) for scene, ids in split.items()}
                for split_name, split in splits.items()
            }
            self.splits = {}
            for split_name, split in splits.items():
                self.splits[split_name] = [
                    (scene, *arg, name)
                    for scene, *arg, name in names
                    if scene in split and int(name.rsplit("_", 1)[0]) in split[scene]
                ]
        else:
            raise ValueError(split_arg)

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.splits[stage],
            self.data[stage],
            self.image_dirs,
            self.tile_managers,
            image_ext=".jpg",
        )

    def dataloader(
        self,
        stage: str,
        shuffle: bool = False,
        num_workers: int = None,
        sampler: Optional[torchdata.Sampler] = None,
    ):
        dataset = self.dataset(stage)
        cfg = self.cfg["loading"][stage]
        batch_size = cfg["batch_size"]

        # Validate batch size is a multiple of views per panorama (n)
        # views_per_panorama = self.num_views // 2
        # views_per_pair = 2
        
        if batch_size % self.num_views != 0:
            raise ValueError(
                f"Batch size ({batch_size}) must be a multiple of views per panorama ({self.num_views})."
            )
        
        num_workers = cfg["num_workers"] if num_workers is None else num_workers

        is_distributed = dist.is_available() and dist.is_initialized()
        if stage == 'train' and is_distributed:
            sampler = DistributedGroupSampler(
                dataset,
                samples_per_group = self.num_views,
                shuffle = True,
            )
            shuffle = False # shuffle of Dataloader should be False when sampler is not None
        else:
            shuffle = shuffle

        loader = torchdata.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)

    def sequence_dataset(self, stage: str, **kwargs):
        keys = self.splits[stage]
        seq2indices = defaultdict(list)
        for index, (_, seq, _) in enumerate(keys):
            seq2indices[seq].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(self.data[stage], indices, **kwargs)
            for i, sub_indices in enumerate(chunks):
                chunk2indices[seq, i] = sub_indices
        # store the index of each chunk in its sequence
        chunk_indices = torch.full((len(keys),), -1)
        for (_, chunk_index), idx in chunk2indices.items():
            chunk_indices[idx] = chunk_index
        self.data[stage]["chunk_index"] = chunk_indices
        dataset = self.dataset(stage)
        return dataset, chunk2indices

    def sequence_dataloader(self, stage: str, shuffle: bool = False, **kwargs):
        dataset, chunk2idx = self.sequence_dataset(stage, **kwargs)
        chunk_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(chunk_keys))
            chunk_keys = [chunk_keys[i] for i in perm]
        key_indices = [i for key in chunk_keys for i in chunk2idx[key]]
        num_workers = self.cfg["loading"][stage]["num_workers"]
        loader = torchdata.DataLoader(
            dataset,
            batch_size=None,
            sampler=key_indices,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return loader, chunk_keys, chunk2idx

class MapillaryPanoDataModule(pl.LightningDataModule):
    dump_filename = "dump.json"
    # images_archive = "images.tar.gz"
    images_dirname = "images/"

    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "mapillary",
        # paths and fetch
        "data_dir": DATASETS_PATH / "MGL_final",
        "local_dir": None,
        "tiles_filename": "tiles.pkl",
        "scenes": "???",
        "split": None,
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 3, "num_workers": 3},
        },
        "filter_for": None,
        "filter_by_ground_angle": None,
        "min_num_points": "???",
    }

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.num_views = 3
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.local_dir = self.cfg.local_dir or os.environ.get("TMPDIR")
        if self.local_dir is not None:
            self.local_dir = Path(self.local_dir, "Sim2Real")
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")

    def prepare_data(self):
        for scene in self.cfg.scenes:
            dump_dir = self.root / scene
            assert (dump_dir / self.dump_filename).exists(), dump_dir
            assert (dump_dir / self.cfg.tiles_filename).exists(), dump_dir
            if self.local_dir is None:
                assert (dump_dir / self.images_dirname).exists(), dump_dir
                continue
            # Cache the folder of images locally to speed up reading
            local_dir = self.local_dir / scene
            if local_dir.exists():
                shutil.rmtree(local_dir)
            local_dir.mkdir(exist_ok=True, parents=True)
            # images_archive = dump_dir / self.images_archive
            # logger.info("Extracting the image archive %s.", images_archive)
            # with tarfile.open(images_archive) as fp:
                # fp.extractall(local_dir)

    def setup(self, stage: Optional[str] = None):
        self.dumps = {}
        self.tile_managers = {}
        self.image_dirs = {}
        names = []

        for scene in self.cfg.scenes:
            logger.info("Loading scene %s.", scene)
            dump_dir = self.root / scene
            logger.info("Loading map tiles %s.", self.cfg.tiles_filename)
            self.tile_managers[scene] = TileManager.load(
                dump_dir / self.cfg.tiles_filename
            )
            groups = self.tile_managers[scene].groups
            if self.cfg.num_classes:  # check consistency
                if set(groups.keys()) != set(self.cfg.num_classes.keys()):
                    raise ValueError(
                        "Inconsistent groups: "
                        f"{groups.keys()} {self.cfg.num_classes.keys()}"
                    )
                for k in groups:
                    if len(groups[k]) != self.cfg.num_classes[k]:
                        raise ValueError(
                            f"{k}: {len(groups[k])} vs {self.cfg.num_classes[k]}"
                        )
            ppm = self.tile_managers[scene].ppm
            if ppm != self.cfg.pixel_per_meter:
                raise ValueError(
                    "The tile manager and the config/model have different ground "
                    f"resolutions: {ppm} vs {self.cfg.pixel_per_meter}"
                )

            logger.info("Loading dump json file %s.", self.dump_filename)
            with (dump_dir / self.dump_filename).open("r") as fp:
                self.dumps[scene] = pack_dump_dict(json.load(fp))
            for seq, per_seq in self.dumps[scene].items():
                for cam_id, cam_dict in per_seq["cameras"].items():
                    if cam_dict["model"] != "PINHOLE":
                        raise ValueError(
                            "Unsupported camera model: "
                            f"{cam_dict['model']} for {scene},{seq},{cam_id}"
                        )

            self.image_dirs[scene] = (
                # (self.local_dir or self.root) / scene / self.images_dirname
                self.root / scene / self.images_dirname
            )
            assert self.image_dirs[scene].exists(), self.image_dirs[scene]

            for seq, data in self.dumps[scene].items():
                for name in data["views"]:
                    names.append((scene, seq, name))

        self.parse_splits(self.cfg.split, names)
        if self.cfg.filter_for is not None:
            self.filter_elements()
        self.pack_data()

    def pack_data(self):
        # We pack the data into compact tensors
        # that can be shared across processes without copy.
        # We group views by panorama to enable panorama-based batching.
        exclude = {
            "compass_angle",
            "compass_accuracy",
            "gps_accuracy",
            "chunk_key",
            "panorama_offset",
        }
        cameras = {
            scene: {seq: per_seq["cameras"] for seq, per_seq in per_scene.items()}
            for scene, per_scene in self.dumps.items()
        }
        points = {
            scene: {
                seq: {
                    i: torch.from_numpy(p) for i, p in per_seq.get("points", {}).items()
                }
                for seq, per_seq in per_scene.items()
            }
            for scene, per_scene in self.dumps.items()
        }
        self.data = {}
        self.panorama_groups = {}  # Store panorama grouping info

        for stage, names in self.splits.items():
            # Group views by panorama ID
            panorama_views = self._group_views_by_panorama(names)

            # Validate that we have complete panoramas (n views each)
            incomplete_panoramas = [
                p_id for p_id, views in panorama_views.items() if len(views) != self.num_views
            ]
            if incomplete_panoramas:
                logger.warning(
                    f"Found {len(incomplete_panoramas)} incomplete panoramas "
                    f"in {stage} set. These will be excluded. "
                    f"First few: {incomplete_panoramas[:5]}"
                )
                # Keep only complete panoramas
                panorama_views = {
                    p_id: views
                    for p_id, views in panorama_views.items()
                    if len(views) == self.num_views
                }

            # Store panorama groups for this stage
            self.panorama_groups[stage] = panorama_views

            # Create flattened names list maintaining panorama view order
            ordered_names = []
            for p_id in sorted(panorama_views.keys()):
                views = panorama_views[p_id]
                # Sort views by suffix for consistent order:
                # suffix_order = {"front": 0, "left": 1, "back": 2, "right": 3}
                # suffix_order = {"view1": 0, "view2": 1, "view3": 2}
                suffix_order = {"view1": 0, "view2": 1, "view3": 2}
                # suffix_order = {"v1": 0, "v2": 1, "v3": 2, "v4": 3, "v5": 4, "v6": 5, "v7": 6, "v8": 7, "v9": 8}

                views_sorted = sorted(
                    views, key=lambda x: suffix_order.get(x[2].split("_")[-1], 999)
                )
                ordered_names.extend(views_sorted)

            # Pack data as before but with grouped panorama ordering
            if ordered_names:
                view = self.dumps[ordered_names[0][0]][ordered_names[0][1]]["views"][
                    ordered_names[0][2]
                ]
                data = {k: [] for k in view.keys() - exclude}
                for scene, seq, name in ordered_names:
                    for k in data:
                        data[k].append(
                            self.dumps[scene][seq]["views"][name].get(k, None)
                        )
                for k in data:
                    v = np.array(data[k])
                    if np.issubdtype(v.dtype, np.integer) or np.issubdtype(
                        v.dtype, np.floating
                    ):
                        v = torch.from_numpy(v)
                    data[k] = v
                data["cameras"] = cameras
                data["points"] = points
                self.data[stage] = data
                self.splits[stage] = np.array(ordered_names)
            else:
                # Empty stage
                self.data[stage] = {"cameras": cameras, "points": points}
                self.splits[stage] = np.array([])

    def _group_views_by_panorama(self, names):
        """Group view names by panorama ID.

        Args:
            names: List of (scene, seq, name) tuples

        Returns:
            Dict mapping panorama_id to list of (scene, seq, name) tuples
        """
        panorama_views = defaultdict(list)
        for scene, seq, name in names:
            # Extract panorama ID from view name (everything before the last underscore)
            if "_" in name:
                panorama_id = name.rsplit("_", 1)[0]
                panorama_views[panorama_id].append((scene, seq, name))
            else:
                # Handle non-panoramic views if any exist
                panorama_views[name].append((scene, seq, name))
        return dict(panorama_views)

    def filter_elements(self):
        for stage, names in self.splits.items():
            names_select = []
            for scene, seq, name in names:
                view = self.dumps[scene][seq]["views"][name]
                if self.cfg.filter_for == "ground_plane":
                    if not (1.0 <= view["height"] <= 3.0):
                        continue
                    planes = self.dumps[scene][seq].get("plane")
                    if planes is not None:
                        inliers = planes[str(view["chunk_id"])][-1]
                        if inliers < 10:
                            continue
                    if self.cfg.filter_by_ground_angle is not None:
                        plane = np.array(view["plane_params"])
                        normal = plane[:3] / np.linalg.norm(plane[:3])
                        angle = np.rad2deg(np.arccos(np.abs(normal[-1])))
                        if angle > self.cfg.filter_by_ground_angle:
                            continue
                elif self.cfg.filter_for == "pointcloud":
                    if len(view["observations"]) < self.cfg.min_num_points:
                        continue
                elif self.cfg.filter_for is not None:
                    raise ValueError(f"Unknown filtering: {self.cfg.filter_for}")
                names_select.append((scene, seq, name))
            logger.info(
                "%s: Keep %d/%d images after filtering for %s.",
                stage,
                len(names_select),
                len(names),
                self.cfg.filter_for,
            )
            self.splits[stage] = names_select

    def parse_splits(self, split_arg, names):
        if split_arg is None:
            self.splits = {
                "train": names,
                "val": names,
            }
        elif isinstance(split_arg, int):
            names = np.random.RandomState(self.cfg.seed).permutation(names).tolist()
            self.splits = {
                "train": names[split_arg:],
                "val": names[:split_arg],
            }
        elif isinstance(split_arg, DictConfig):
            scenes_val = set(split_arg.val)
            scenes_train = set(split_arg.train)
            assert len(scenes_val - set(self.cfg.scenes)) == 0
            assert len(scenes_train - set(self.cfg.scenes)) == 0
            self.splits = {
                "train": [n for n in names if n[0] in scenes_train],
                "val": [n for n in names if n[0] in scenes_val],
            }
        elif isinstance(split_arg, str):
            if (self.root / split_arg).exists():
                # Common split file.
                with (self.root / split_arg).open("r") as fp:
                    splits = json.load(fp)
            else:
                # Per-scene split file.
                splits = defaultdict(dict)
                for scene in self.cfg.scenes:
                    with (self.root / split_arg.format(scene=scene)).open("r") as fp:
                        scene_splits = json.load(fp)
                    for split_name in scene_splits:
                        splits[split_name][scene] = scene_splits[split_name]
            splits = {
                split_name: {scene: set(ids) for scene, ids in split.items()}
                for split_name, split in splits.items()
            }
            self.splits = {}
            for split_name, split in splits.items():
                self.splits[split_name] = [
                    (scene, *arg, name)
                    for scene, *arg, name in names
                    if scene in split and int(name.rsplit("_", 1)[0]) in split[scene]
                ]
        else:
            raise ValueError(split_arg)

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.splits[stage],
            self.data[stage],
            self.image_dirs,
            self.tile_managers,
            image_ext=".jpg",
        )

    def dataloader(
        self,
        stage: str,
        shuffle: bool = False,
        num_workers: int = None,
        sampler: Optional[torchdata.Sampler] = None,
    ):
        dataset = self.dataset(stage)
        cfg = self.cfg["loading"][stage]
        batch_size = cfg["batch_size"]

        # Validate batch size is a multiple of views per panorama (n)
        # views_per_panorama = self.num_views // 2
        # views_per_pair = 2
        
        if batch_size % self.num_views != 0:
            raise ValueError(
                f"Batch size ({batch_size}) must be a multiple of views per panorama ({self.num_views})."
            )
        
        num_workers = cfg["num_workers"] if num_workers is None else num_workers

        is_distributed = dist.is_available() and dist.is_initialized()
        if stage == 'train' and is_distributed:
            sampler = DistributedGroupSampler(
                dataset,
                samples_per_group = self.num_views,
                shuffle = True,
            )
            shuffle = False # shuffle of Dataloader should be False when sampler is not None
        else:
            shuffle = shuffle

        loader = torchdata.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)

    def sequence_dataset(self, stage: str, **kwargs):
        keys = self.splits[stage]
        seq2indices = defaultdict(list)
        for index, (_, seq, _) in enumerate(keys):
            seq2indices[seq].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(self.data[stage], indices, **kwargs)
            for i, sub_indices in enumerate(chunks):
                chunk2indices[seq, i] = sub_indices
        # store the index of each chunk in its sequence
        chunk_indices = torch.full((len(keys),), -1)
        for (_, chunk_index), idx in chunk2indices.items():
            chunk_indices[idx] = chunk_index
        self.data[stage]["chunk_index"] = chunk_indices
        dataset = self.dataset(stage)
        return dataset, chunk2indices

    def sequence_dataloader(self, stage: str, shuffle: bool = False, **kwargs):
        dataset, chunk2idx = self.sequence_dataset(stage, **kwargs)
        chunk_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(chunk_keys))
            chunk_keys = [chunk_keys[i] for i in perm]
        key_indices = [i for key in chunk_keys for i in chunk2idx[key]]
        num_workers = self.cfg["loading"][stage]["num_workers"]
        loader = torchdata.DataLoader(
            dataset,
            batch_size=None,
            sampler=key_indices,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return loader, chunk_keys, chunk2idx

class MapillaryPanoMixedDataModule(pl.LightningDataModule):
    dump_filename = "dump.json"
    # images_archive = "images.tar.gz"
    images_dirname = "images/"

    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "mapillary",
        # paths and fetch
        "data_dir": DATASETS_PATH,
        "local_dir": None,
        "tiles_filename": "tiles.pkl",
        "scenes": "???",
        "split": None,
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 3, "num_workers": 0},
        },
        "filter_for": None,
        "filter_by_ground_angle": None,
        "min_num_points": "???",
    }

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.num_views = 3
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.local_dir = self.cfg.local_dir or os.environ.get("TMPDIR") 
        if self.local_dir is not None:
            self.local_dir = Path(self.local_dir , "OrienterNet/datasets")
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")

    def prepare_data(self):
        for scene in self.cfg.scenes:
            dump_dir = self.root / scene
            assert (dump_dir / self.dump_filename).exists(), dump_dir
            assert (dump_dir / self.cfg.tiles_filename).exists(), dump_dir
            if self.local_dir is None:
                assert (dump_dir / self.images_dirname).exists(), dump_dir
                continue
            # Cache the folder of images locally to speed up reading
            local_dir = self.local_dir / scene
            if local_dir.exists():
                shutil.rmtree(local_dir)
            local_dir.mkdir(exist_ok=True, parents=True)
            # images_archive = dump_dir / self.images_archive
            # logger.info("Extracting the image archive %s.", images_archive)
            # with tarfile.open(images_archive) as fp:
                # fp.extractall(local_dir)

    def setup(self, stage: Optional[str] = None):
        self.dumps = {}
        self.tile_managers = {}
        self.image_dirs = {}
        names = []

        for scene in self.cfg.scenes:
            logger.info("Loading scene %s.", scene)
            dump_dir = self.root / scene
            logger.info("Loading map tiles %s.", self.cfg.tiles_filename)
            self.tile_managers[scene] = TileManager.load(
                dump_dir / self.cfg.tiles_filename
            )
            groups = self.tile_managers[scene].groups
            if self.cfg.num_classes:  # check consistency
                if set(groups.keys()) != set(self.cfg.num_classes.keys()):
                    raise ValueError(
                        "Inconsistent groups: "
                        f"{groups.keys()} {self.cfg.num_classes.keys()}"
                    )
                for k in groups:
                    if len(groups[k]) != self.cfg.num_classes[k]:
                        raise ValueError(
                            f"{k}: {len(groups[k])} vs {self.cfg.num_classes[k]}"
                        )
            ppm = self.tile_managers[scene].ppm
            if ppm != self.cfg.pixel_per_meter:
                raise ValueError(
                    "The tile manager and the config/model have different ground "
                    f"resolutions: {ppm} vs {self.cfg.pixel_per_meter}"
                )

            logger.info("Loading dump json file %s.", self.dump_filename)
            with (dump_dir / self.dump_filename).open("r") as fp:
                self.dumps[scene] = pack_dump_dict(json.load(fp))
            for seq, per_seq in self.dumps[scene].items():
                for cam_id, cam_dict in per_seq["cameras"].items():
                    if cam_dict["model"] != "PINHOLE":
                        raise ValueError(
                            "Unsupported camera model: "
                            f"{cam_dict['model']} for {scene},{seq},{cam_id}"
                        )

            self.image_dirs[scene] = (
                # (self.local_dir or self.root) / scene / self.images_dirname
                self.root / scene / self.images_dirname
            )
            assert self.image_dirs[scene].exists(), self.image_dirs[scene]

            for seq, data in self.dumps[scene].items():
                for name in data["views"]:
                    names.append((scene, seq, name))

        self.parse_splits(self.cfg.split, names)
        if self.cfg.filter_for is not None:
            self.filter_elements()
        self.pack_data()

    def pack_data(self):
        # We pack the data into compact tensors
        # that can be shared across processes without copy.
        # We group views by panorama to enable panorama-based batching.
        exclude = {
            "compass_angle",
            "compass_accuracy",
            "gps_accuracy",
            "chunk_key",
            "panorama_offset",
        }
        cameras = {
            scene: {seq: per_seq["cameras"] for seq, per_seq in per_scene.items()}
            for scene, per_scene in self.dumps.items()
        }
        points = {
            scene: {
                seq: {
                    i: torch.from_numpy(p) for i, p in per_seq.get("points", {}).items()
                }
                for seq, per_seq in per_scene.items()
            }
            for scene, per_scene in self.dumps.items()
        }
        self.data = {}
        self.panorama_groups = {}  # Store panorama grouping info

        for stage, names in self.splits.items():
            # Group views by panorama ID
            panorama_views = self._group_views_by_panorama(names)

            # Validate that we have complete panoramas (n views each)
            incomplete_panoramas = [
                p_id for p_id, views in panorama_views.items() if len(views) != self.num_views
            ]
            if incomplete_panoramas:
                logger.warning(
                    f"Found {len(incomplete_panoramas)} incomplete panoramas "
                    f"in {stage} set. These will be excluded. "
                    f"First few: {incomplete_panoramas[:5]}"
                )
                # Keep only complete panoramas
                panorama_views = {
                    p_id: views
                    for p_id, views in panorama_views.items()
                    if len(views) == self.num_views
                }

            # Store panorama groups for this stage
            self.panorama_groups[stage] = panorama_views

            # Create flattened names list maintaining panorama view order
            ordered_names = []
            for p_id in sorted(panorama_views.keys()):
                views = panorama_views[p_id]
                # Sort views by suffix for consistent order:
                # suffix_order = {"front": 0, "left": 1, "back": 2, "right": 3}
                # suffix_order = {"view1": 0, "view2": 1, "view3": 2}
                suffix_order = {"view1": 0, "view2": 1, "view3": 2}
                # suffix_order = {"v1": 0, "v2": 1, "v3": 2, "v4": 3, "v5": 4, "v6": 5, "v7": 6, "v8": 7, "v9": 8}

                views_sorted = sorted(
                    views, key=lambda x: suffix_order.get(x[2].split("_")[-1], 999)
                )
                ordered_names.extend(views_sorted)

            # Pack data as before but with grouped panorama ordering
            if ordered_names:
                view = self.dumps[ordered_names[0][0]][ordered_names[0][1]]["views"][
                    ordered_names[0][2]
                ]
                data = {k: [] for k in view.keys() - exclude}
                for scene, seq, name in ordered_names:
                    for k in data:
                        data[k].append(
                            self.dumps[scene][seq]["views"][name].get(k, None)
                        )
                for k in data:
                    v = np.array(data[k])
                    if np.issubdtype(v.dtype, np.integer) or np.issubdtype(
                        v.dtype, np.floating
                    ):
                        v = torch.from_numpy(v)
                    data[k] = v
                data["cameras"] = cameras
                data["points"] = points
                self.data[stage] = data
                self.splits[stage] = np.array(ordered_names)
            else:
                # Empty stage
                self.data[stage] = {"cameras": cameras, "points": points}
                self.splits[stage] = np.array([])

    def _group_views_by_panorama(self, names):
        """Group view names by panorama ID.

        Args:
            names: List of (scene, seq, name) tuples

        Returns:
            Dict mapping panorama_id to list of (scene, seq, name) tuples
        """
        panorama_views = defaultdict(list)
        for scene, seq, name in names:
            # Extract panorama ID from view name (everything before the last underscore)
            if "_" in name:
                panorama_id = scene + name.rsplit("_", 1)[0]
                panorama_views[panorama_id].append((scene, seq, name))
            else:
                # Handle non-panoramic views if any exist
                panorama_views[name].append((scene, seq, name))
        return dict(panorama_views)

    def filter_elements(self):
        for stage, names in self.splits.items():
            names_select = []
            for scene, seq, name in names:
                view = self.dumps[scene][seq]["views"][name]
                if self.cfg.filter_for == "ground_plane":
                    if not (1.0 <= view["height"] <= 3.0):
                        continue
                    planes = self.dumps[scene][seq].get("plane")
                    if planes is not None:
                        inliers = planes[str(view["chunk_id"])][-1]
                        if inliers < 10:
                            continue
                    if self.cfg.filter_by_ground_angle is not None:
                        plane = np.array(view["plane_params"])
                        normal = plane[:3] / np.linalg.norm(plane[:3])
                        angle = np.rad2deg(np.arccos(np.abs(normal[-1])))
                        if angle > self.cfg.filter_by_ground_angle:
                            continue
                elif self.cfg.filter_for == "pointcloud":
                    if len(view["observations"]) < self.cfg.min_num_points:
                        continue
                elif self.cfg.filter_for is not None:
                    raise ValueError(f"Unknown filtering: {self.cfg.filter_for}")
                names_select.append((scene, seq, name))
            logger.info(
                "%s: Keep %d/%d images after filtering for %s.",
                stage,
                len(names_select),
                len(names),
                self.cfg.filter_for,
            )
            self.splits[stage] = names_select

    def parse_splits(self, split_arg, names):
        if split_arg is None:
            self.splits = {
                "train": names,
                "val": names,
            }
        elif isinstance(split_arg, int):
            names = np.random.RandomState(self.cfg.seed).permutation(names).tolist()
            self.splits = {
                "train": names[split_arg:],
                "val": names[:split_arg],
            }
        elif isinstance(split_arg, DictConfig):
            scenes_val = set(split_arg.val)
            scenes_train = set(split_arg.train)
            assert len(scenes_val - set(self.cfg.scenes)) == 0
            assert len(scenes_train - set(self.cfg.scenes)) == 0
            self.splits = {
                "train": [n for n in names if n[0] in scenes_train],
                "val": [n for n in names if n[0] in scenes_val],
            }
        elif isinstance(split_arg, str):
            if (self.root / split_arg).exists():
                # Common split file.
                with (self.root / split_arg).open("r") as fp:
                    splits = json.load(fp)
            else:
                # Per-scene split file.
                splits = defaultdict(dict)
                for scene in self.cfg.scenes:
                    with (self.root / split_arg.format(scene=scene)).open("r") as fp:
                        scene_splits = json.load(fp)
                    for split_name in scene_splits:
                        splits[split_name][scene] = scene_splits[split_name]
            splits = {
                split_name: {scene: set(ids) for scene, ids in split.items()}
                for split_name, split in splits.items()
            }
            self.splits = {}
            for split_name, split in splits.items():
                self.splits[split_name] = [
                    (scene, *arg, name)
                    for scene, *arg, name in names
                    if scene in split and int(name.rsplit("_", 1)[0]) in split[scene]
                ]
        else:
            raise ValueError(split_arg)

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.splits[stage],
            self.data[stage],
            self.image_dirs,
            self.tile_managers,
            image_ext=".jpg",
        )

    def dataloader(
        self,
        stage: str,
        shuffle: bool = False,
        num_workers: int = None,
        sampler: Optional[torchdata.Sampler] = None,
    ):
        dataset = self.dataset(stage)
        cfg = self.cfg["loading"][stage]
        batch_size = cfg["batch_size"]

        # Validate batch size is a multiple of views per panorama (n)
        # views_per_panorama = self.num_views // 2
        # views_per_pair = 2
        
        if batch_size % self.num_views != 0:
            raise ValueError(
                f"Batch size ({batch_size}) must be a multiple of views per panorama ({self.num_views})."
            )
        
        num_workers = cfg["num_workers"] if num_workers is None else num_workers

        is_distributed = dist.is_available() and dist.is_initialized()
        if stage == 'train' and is_distributed:
            sampler = DistributedGroupSampler(
                dataset,
                samples_per_group = self.num_views,
                shuffle = True,
            )
            shuffle = False # shuffle of Dataloader should be False when sampler is not None
        else:
            shuffle = shuffle

        loader = torchdata.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)

    def sequence_dataset(self, stage: str, **kwargs):
        keys = self.splits[stage]
        seq2indices = defaultdict(list)
        for index, (_, seq, _) in enumerate(keys):
            seq2indices[seq].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(self.data[stage], indices, **kwargs)
            for i, sub_indices in enumerate(chunks):
                chunk2indices[seq, i] = sub_indices
        # store the index of each chunk in its sequence
        chunk_indices = torch.full((len(keys),), -1)
        for (_, chunk_index), idx in chunk2indices.items():
            chunk_indices[idx] = chunk_index
        self.data[stage]["chunk_index"] = chunk_indices
        dataset = self.dataset(stage)
        return dataset, chunk2indices

    def sequence_dataloader(self, stage: str, shuffle: bool = False, **kwargs):
        dataset, chunk2idx = self.sequence_dataset(stage, **kwargs)
        chunk_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(chunk_keys))
            chunk_keys = [chunk_keys[i] for i in perm]
        key_indices = [i for key in chunk_keys for i in chunk2idx[key]]
        num_workers = self.cfg["loading"][stage]["num_workers"]
        loader = torchdata.DataLoader(
            dataset,
            batch_size=None,
            sampler=key_indices,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return loader, chunk_keys, chunk2idx

class MapillaryPanoRainyDataModule(pl.LightningDataModule):
    dump_filename = "dump.json"
    # images_archive = "images.tar.gz"
    images_dirname = "images/"

    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "mapillary",
        # paths and fetch
        "data_dir": DATASETS_PATH / "MGL_rainy",
        "clean_data_dir": DATASETS_PATH / "MGL_final",
        "local_dir": None,
        "tiles_filename": "tiles.pkl",
        "scenes": "???",
        "split": None,
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 3, "num_workers": 3},
        },
        "filter_for": None,
        "filter_by_ground_angle": None,
        "min_num_points": "???",
    }

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.num_views = 3
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.clean_root = Path(self.cfg.clean_data_dir)
        self.local_dir = self.cfg.local_dir or os.environ.get("TMPDIR")
        if self.local_dir is not None:
            self.local_dir = Path(self.local_dir, "MGL_rainy")
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")

    def prepare_data(self):
        for scene in self.cfg.scenes:
            dump_dir = self.clean_root / scene
            assert (dump_dir / self.dump_filename).exists(), dump_dir
            assert (dump_dir / self.cfg.tiles_filename).exists(), dump_dir
            if self.local_dir is None:
                assert (dump_dir / self.images_dirname).exists(), dump_dir
                continue
            # Cache the folder of images locally to speed up reading
            local_dir = self.local_dir / scene
            if local_dir.exists():
                shutil.rmtree(local_dir)
            local_dir.mkdir(exist_ok=True, parents=True)
            # images_archive = dump_dir / self.images_archive
            # logger.info("Extracting the image archive %s.", images_archive)
            # with tarfile.open(images_archive) as fp:
                # fp.extractall(local_dir)

    def setup(self, stage: Optional[str] = None):
        self.dumps = {}
        self.tile_managers = {}
        self.image_dirs = {}
        names = []

        for scene in self.cfg.scenes:
            logger.info("Loading scene %s.", scene)
            dump_dir = self.clean_root / scene
            logger.info("Loading map tiles %s.", self.cfg.tiles_filename)
            self.tile_managers[scene] = TileManager.load(
                dump_dir / self.cfg.tiles_filename
            )
            groups = self.tile_managers[scene].groups
            if self.cfg.num_classes:  # check consistency
                if set(groups.keys()) != set(self.cfg.num_classes.keys()):
                    raise ValueError(
                        "Inconsistent groups: "
                        f"{groups.keys()} {self.cfg.num_classes.keys()}"
                    )
                for k in groups:
                    if len(groups[k]) != self.cfg.num_classes[k]:
                        raise ValueError(
                            f"{k}: {len(groups[k])} vs {self.cfg.num_classes[k]}"
                        )
            ppm = self.tile_managers[scene].ppm
            if ppm != self.cfg.pixel_per_meter:
                raise ValueError(
                    "The tile manager and the config/model have different ground "
                    f"resolutions: {ppm} vs {self.cfg.pixel_per_meter}"
                )

            logger.info("Loading dump json file %s.", self.dump_filename)
            with (dump_dir / self.dump_filename).open("r") as fp:
                self.dumps[scene] = pack_dump_dict(json.load(fp))
            for seq, per_seq in self.dumps[scene].items():
                for cam_id, cam_dict in per_seq["cameras"].items():
                    if cam_dict["model"] != "PINHOLE":
                        raise ValueError(
                            "Unsupported camera model: "
                            f"{cam_dict['model']} for {scene},{seq},{cam_id}"
                        )

            self.image_dirs[scene] = (
                # (self.local_dir or self.root) / scene / self.images_dirname
                self.root / scene / self.images_dirname
            )
            assert self.image_dirs[scene].exists(), self.image_dirs[scene]

            for seq, data in self.dumps[scene].items():
                for name in data["views"]:
                    names.append((scene, seq, name))

        self.parse_splits(self.cfg.split, names)
        if self.cfg.filter_for is not None:
            self.filter_elements()
        self.pack_data()

    def pack_data(self):
        # We pack the data into compact tensors
        # that can be shared across processes without copy.
        # We group views by panorama to enable panorama-based batching.
        exclude = {
            "compass_angle",
            "compass_accuracy",
            "gps_accuracy",
            "chunk_key",
            "panorama_offset",
        }
        cameras = {
            scene: {seq: per_seq["cameras"] for seq, per_seq in per_scene.items()}
            for scene, per_scene in self.dumps.items()
        }
        points = {
            scene: {
                seq: {
                    i: torch.from_numpy(p) for i, p in per_seq.get("points", {}).items()
                }
                for seq, per_seq in per_scene.items()
            }
            for scene, per_scene in self.dumps.items()
        }
        self.data = {}
        self.panorama_groups = {}  # Store panorama grouping info

        for stage, names in self.splits.items():
            # Group views by panorama ID
            panorama_views = self._group_views_by_panorama(names)

            # Validate that we have complete panoramas (n views each)
            incomplete_panoramas = [
                p_id for p_id, views in panorama_views.items() if len(views) != self.num_views
            ]
            if incomplete_panoramas:
                logger.warning(
                    f"Found {len(incomplete_panoramas)} incomplete panoramas "
                    f"in {stage} set. These will be excluded. "
                    f"First few: {incomplete_panoramas[:5]}"
                )
                # Keep only complete panoramas
                panorama_views = {
                    p_id: views
                    for p_id, views in panorama_views.items()
                    if len(views) == self.num_views
                }

            # Store panorama groups for this stage
            self.panorama_groups[stage] = panorama_views

            # Create flattened names list maintaining panorama view order
            ordered_names = []
            for p_id in sorted(panorama_views.keys()):
                views = panorama_views[p_id]
                # Sort views by suffix for consistent order:
                # suffix_order = {"front": 0, "left": 1, "back": 2, "right": 3}
                # suffix_order = {"view1": 0, "view2": 1, "view3": 2}
                suffix_order = {"view1": 0, "view2": 1, "view3": 2}
                # suffix_order = {"v1": 0, "v2": 1, "v3": 2, "v4": 3, "v5": 4, "v6": 5, "v7": 6, "v8": 7, "v9": 8}

                views_sorted = sorted(
                    views, key=lambda x: suffix_order.get(x[2].split("_")[-1], 999)
                )
                ordered_names.extend(views_sorted)

            # Pack data as before but with grouped panorama ordering
            if ordered_names:
                view = self.dumps[ordered_names[0][0]][ordered_names[0][1]]["views"][
                    ordered_names[0][2]
                ]
                data = {k: [] for k in view.keys() - exclude}
                for scene, seq, name in ordered_names:
                    for k in data:
                        data[k].append(
                            self.dumps[scene][seq]["views"][name].get(k, None)
                        )
                for k in data:
                    v = np.array(data[k])
                    if np.issubdtype(v.dtype, np.integer) or np.issubdtype(
                        v.dtype, np.floating
                    ):
                        v = torch.from_numpy(v)
                    data[k] = v
                data["cameras"] = cameras
                data["points"] = points
                self.data[stage] = data
                self.splits[stage] = np.array(ordered_names)
            else:
                # Empty stage
                self.data[stage] = {"cameras": cameras, "points": points}
                self.splits[stage] = np.array([])

    def _group_views_by_panorama(self, names):
        """Group view names by panorama ID.

        Args:
            names: List of (scene, seq, name) tuples

        Returns:
            Dict mapping panorama_id to list of (scene, seq, name) tuples
        """
        panorama_views = defaultdict(list)
        for scene, seq, name in names:
            # Extract panorama ID from view name (everything before the last underscore)
            if "_" in name:
                panorama_id = name.rsplit("_", 1)[0]
                panorama_views[panorama_id].append((scene, seq, name))
            else:
                # Handle non-panoramic views if any exist
                panorama_views[name].append((scene, seq, name))
        return dict(panorama_views)

    def filter_elements(self):
        for stage, names in self.splits.items():
            names_select = []
            for scene, seq, name in names:
                view = self.dumps[scene][seq]["views"][name]
                if self.cfg.filter_for == "ground_plane":
                    if not (1.0 <= view["height"] <= 3.0):
                        continue
                    planes = self.dumps[scene][seq].get("plane")
                    if planes is not None:
                        inliers = planes[str(view["chunk_id"])][-1]
                        if inliers < 10:
                            continue
                    if self.cfg.filter_by_ground_angle is not None:
                        plane = np.array(view["plane_params"])
                        normal = plane[:3] / np.linalg.norm(plane[:3])
                        angle = np.rad2deg(np.arccos(np.abs(normal[-1])))
                        if angle > self.cfg.filter_by_ground_angle:
                            continue
                elif self.cfg.filter_for == "pointcloud":
                    if len(view["observations"]) < self.cfg.min_num_points:
                        continue
                elif self.cfg.filter_for is not None:
                    raise ValueError(f"Unknown filtering: {self.cfg.filter_for}")
                names_select.append((scene, seq, name))
            logger.info(
                "%s: Keep %d/%d images after filtering for %s.",
                stage,
                len(names_select),
                len(names),
                self.cfg.filter_for,
            )
            self.splits[stage] = names_select

    def parse_splits(self, split_arg, names):
        if split_arg is None:
            self.splits = {
                "train": names,
                "val": names,
            }
        elif isinstance(split_arg, int):
            names = np.random.RandomState(self.cfg.seed).permutation(names).tolist()
            self.splits = {
                "train": names[split_arg:],
                "val": names[:split_arg],
            }
        elif isinstance(split_arg, DictConfig):
            scenes_val = set(split_arg.val)
            scenes_train = set(split_arg.train)
            assert len(scenes_val - set(self.cfg.scenes)) == 0
            assert len(scenes_train - set(self.cfg.scenes)) == 0
            self.splits = {
                "train": [n for n in names if n[0] in scenes_train],
                "val": [n for n in names if n[0] in scenes_val],
            }
        elif isinstance(split_arg, str):
            if (self.root / split_arg).exists():
                # Common split file.
                with (self.root / split_arg).open("r") as fp:
                    splits = json.load(fp)
            else:
                # Per-scene split file.
                splits = defaultdict(dict)
                for scene in self.cfg.scenes:
                    with (self.root / split_arg.format(scene=scene)).open("r") as fp:
                        scene_splits = json.load(fp)
                    for split_name in scene_splits:
                        splits[split_name][scene] = scene_splits[split_name]
            splits = {
                split_name: {scene: set(ids) for scene, ids in split.items()}
                for split_name, split in splits.items()
            }
            self.splits = {}
            for split_name, split in splits.items():
                self.splits[split_name] = [
                    (scene, *arg, name)
                    for scene, *arg, name in names
                    if scene in split and int(name.rsplit("_", 1)[0]) in split[scene]
                ]
        else:
            raise ValueError(split_arg)

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.splits[stage],
            self.data[stage],
            self.image_dirs,
            self.tile_managers,
            image_ext=".jpg",
        )

    def dataloader(
        self,
        stage: str,
        shuffle: bool = False,
        num_workers: int = None,
        sampler: Optional[torchdata.Sampler] = None,
    ):
        dataset = self.dataset(stage)
        cfg = self.cfg["loading"][stage]
        batch_size = cfg["batch_size"]

        # Validate batch size is a multiple of views per panorama (n)
        # views_per_panorama = self.num_views // 2
        # views_per_pair = 2
        
        if batch_size % self.num_views != 0:
            raise ValueError(
                f"Batch size ({batch_size}) must be a multiple of views per panorama ({self.num_views})."
            )
        
        num_workers = cfg["num_workers"] if num_workers is None else num_workers

        is_distributed = dist.is_available() and dist.is_initialized()
        if stage == 'train' and is_distributed:
            sampler = DistributedGroupSampler(
                dataset,
                samples_per_group = self.num_views,
                shuffle = True,
            )
            shuffle = False # shuffle of Dataloader should be False when sampler is not None
        else:
            shuffle = shuffle

        loader = torchdata.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)

    def sequence_dataset(self, stage: str, **kwargs):
        keys = self.splits[stage]
        seq2indices = defaultdict(list)
        for index, (_, seq, _) in enumerate(keys):
            seq2indices[seq].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(self.data[stage], indices, **kwargs)
            for i, sub_indices in enumerate(chunks):
                chunk2indices[seq, i] = sub_indices
        # store the index of each chunk in its sequence
        chunk_indices = torch.full((len(keys),), -1)
        for (_, chunk_index), idx in chunk2indices.items():
            chunk_indices[idx] = chunk_index
        self.data[stage]["chunk_index"] = chunk_indices
        dataset = self.dataset(stage)
        return dataset, chunk2indices

    def sequence_dataloader(self, stage: str, shuffle: bool = False, **kwargs):
        dataset, chunk2idx = self.sequence_dataset(stage, **kwargs)
        chunk_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(chunk_keys))
            chunk_keys = [chunk_keys[i] for i in perm]
        key_indices = [i for key in chunk_keys for i in chunk2idx[key]]
        num_workers = self.cfg["loading"][stage]["num_workers"]
        loader = torchdata.DataLoader(
            dataset,
            batch_size=None,
            sampler=key_indices,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return loader, chunk_keys, chunk2idx

class MapillaryPanoNightDataModule(pl.LightningDataModule):
    dump_filename = "dump.json"
    # images_archive = "images.tar.gz"
    images_dirname = "images/"

    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "mapillary",
        # paths and fetch
        "data_dir": DATASETS_PATH / "MGL_final",
        "noise_data_dir": DATASETS_PATH / "MGL_night",
        "local_dir": None,
        "tiles_filename": "tiles.pkl",
        "scenes": "???",
        "split": None,
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 3, "num_workers": 3},
        },
        "filter_for": None,
        "filter_by_ground_angle": None,
        "min_num_points": "???",
    }

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.num_views = 3
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.noise_root = Path(self.cfg.noise_data_dir)
        self.local_dir = self.cfg.local_dir or os.environ.get("TMPDIR")
        if self.local_dir is not None:
            self.local_dir = Path(self.local_dir, "MGL_night")
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")

    def prepare_data(self):
        for scene in self.cfg.scenes:
            dump_dir = self.root / scene
            assert (dump_dir / self.dump_filename).exists(), dump_dir
            assert (dump_dir / self.cfg.tiles_filename).exists(), dump_dir
            if self.local_dir is None:
                assert (self.noise_root / scene / self.images_dirname).exists(), dump_dir
                continue
            # Cache the folder of images locally to speed up reading
            local_dir = self.local_dir / scene
            if local_dir.exists():
                shutil.rmtree(local_dir)
            local_dir.mkdir(exist_ok=True, parents=True)
            # images_archive = dump_dir / self.images_archive
            # logger.info("Extracting the image archive %s.", images_archive)
            # with tarfile.open(images_archive) as fp:
                # fp.extractall(local_dir)

    def setup(self, stage: Optional[str] = None):
        self.dumps = {}
        self.tile_managers = {}
        self.image_dirs = {}
        names = []

        for scene in self.cfg.scenes:
            logger.info("Loading scene %s.", scene)
            dump_dir = self.root / scene
            logger.info("Loading map tiles %s.", self.cfg.tiles_filename)
            self.tile_managers[scene] = TileManager.load(
                dump_dir / self.cfg.tiles_filename
            )
            groups = self.tile_managers[scene].groups
            if self.cfg.num_classes:  # check consistency
                if set(groups.keys()) != set(self.cfg.num_classes.keys()):
                    raise ValueError(
                        "Inconsistent groups: "
                        f"{groups.keys()} {self.cfg.num_classes.keys()}"
                    )
                for k in groups:
                    if len(groups[k]) != self.cfg.num_classes[k]:
                        raise ValueError(
                            f"{k}: {len(groups[k])} vs {self.cfg.num_classes[k]}"
                        )
            ppm = self.tile_managers[scene].ppm
            if ppm != self.cfg.pixel_per_meter:
                raise ValueError(
                    "The tile manager and the config/model have different ground "
                    f"resolutions: {ppm} vs {self.cfg.pixel_per_meter}"
                )

            logger.info("Loading dump json file %s.", self.dump_filename)
            with (dump_dir / self.dump_filename).open("r") as fp:
                self.dumps[scene] = pack_dump_dict(json.load(fp))
            for seq, per_seq in self.dumps[scene].items():
                for cam_id, cam_dict in per_seq["cameras"].items():
                    if cam_dict["model"] != "PINHOLE":
                        raise ValueError(
                            "Unsupported camera model: "
                            f"{cam_dict['model']} for {scene},{seq},{cam_id}"
                        )

            self.image_dirs[scene] = (
                # (self.local_dir or self.root) / scene / self.images_dirname
                self.noise_root / scene / self.images_dirname
            )
            assert self.image_dirs[scene].exists(), self.image_dirs[scene]

            for seq, data in self.dumps[scene].items():
                for name in data["views"]:
                    names.append((scene, seq, name))

        self.parse_splits(self.cfg.split, names)
        if self.cfg.filter_for is not None:
            self.filter_elements()
        self.pack_data()

    def pack_data(self):
        # We pack the data into compact tensors
        # that can be shared across processes without copy.
        # We group views by panorama to enable panorama-based batching.
        exclude = {
            "compass_angle",
            "compass_accuracy",
            "gps_accuracy",
            "chunk_key",
            "panorama_offset",
        }
        cameras = {
            scene: {seq: per_seq["cameras"] for seq, per_seq in per_scene.items()}
            for scene, per_scene in self.dumps.items()
        }
        points = {
            scene: {
                seq: {
                    i: torch.from_numpy(p) for i, p in per_seq.get("points", {}).items()
                }
                for seq, per_seq in per_scene.items()
            }
            for scene, per_scene in self.dumps.items()
        }
        self.data = {}
        self.panorama_groups = {}  # Store panorama grouping info

        for stage, names in self.splits.items():
            # Group views by panorama ID
            panorama_views = self._group_views_by_panorama(names)

            # Validate that we have complete panoramas (n views each)
            incomplete_panoramas = [
                p_id for p_id, views in panorama_views.items() if len(views) != self.num_views
            ]
            if incomplete_panoramas:
                logger.warning(
                    f"Found {len(incomplete_panoramas)} incomplete panoramas "
                    f"in {stage} set. These will be excluded. "
                    f"First few: {incomplete_panoramas[:5]}"
                )
                # Keep only complete panoramas
                panorama_views = {
                    p_id: views
                    for p_id, views in panorama_views.items()
                    if len(views) == self.num_views
                }

            # Store panorama groups for this stage
            self.panorama_groups[stage] = panorama_views

            # Create flattened names list maintaining panorama view order
            ordered_names = []
            for p_id in sorted(panorama_views.keys()):
                views = panorama_views[p_id]
                # Sort views by suffix for consistent order:
                # suffix_order = {"front": 0, "left": 1, "back": 2, "right": 3}
                # suffix_order = {"view1": 0, "view2": 1, "view3": 2}
                suffix_order = {"view1": 0, "view2": 1, "view3": 2}
                # suffix_order = {"v1": 0, "v2": 1, "v3": 2, "v4": 3, "v5": 4, "v6": 5, "v7": 6, "v8": 7, "v9": 8}

                views_sorted = sorted(
                    views, key=lambda x: suffix_order.get(x[2].split("_")[-1], 999)
                )
                ordered_names.extend(views_sorted)

            # Pack data as before but with grouped panorama ordering
            if ordered_names:
                view = self.dumps[ordered_names[0][0]][ordered_names[0][1]]["views"][
                    ordered_names[0][2]
                ]
                data = {k: [] for k in view.keys() - exclude}
                for scene, seq, name in ordered_names:
                    for k in data:
                        data[k].append(
                            self.dumps[scene][seq]["views"][name].get(k, None)
                        )
                for k in data:
                    v = np.array(data[k])
                    if np.issubdtype(v.dtype, np.integer) or np.issubdtype(
                        v.dtype, np.floating
                    ):
                        v = torch.from_numpy(v)
                    data[k] = v
                data["cameras"] = cameras
                data["points"] = points
                self.data[stage] = data
                self.splits[stage] = np.array(ordered_names)
            else:
                # Empty stage
                self.data[stage] = {"cameras": cameras, "points": points}
                self.splits[stage] = np.array([])

    def _group_views_by_panorama(self, names):
        """Group view names by panorama ID.

        Args:
            names: List of (scene, seq, name) tuples

        Returns:
            Dict mapping panorama_id to list of (scene, seq, name) tuples
        """
        panorama_views = defaultdict(list)
        for scene, seq, name in names:
            # Extract panorama ID from view name (everything before the last underscore)
            if "_" in name:
                panorama_id = name.rsplit("_", 1)[0]
                panorama_views[panorama_id].append((scene, seq, name))
            else:
                # Handle non-panoramic views if any exist
                panorama_views[name].append((scene, seq, name))
        return dict(panorama_views)

    def filter_elements(self):
        for stage, names in self.splits.items():
            names_select = []
            for scene, seq, name in names:
                view = self.dumps[scene][seq]["views"][name]
                if self.cfg.filter_for == "ground_plane":
                    if not (1.0 <= view["height"] <= 3.0):
                        continue
                    planes = self.dumps[scene][seq].get("plane")
                    if planes is not None:
                        inliers = planes[str(view["chunk_id"])][-1]
                        if inliers < 10:
                            continue
                    if self.cfg.filter_by_ground_angle is not None:
                        plane = np.array(view["plane_params"])
                        normal = plane[:3] / np.linalg.norm(plane[:3])
                        angle = np.rad2deg(np.arccos(np.abs(normal[-1])))
                        if angle > self.cfg.filter_by_ground_angle:
                            continue
                elif self.cfg.filter_for == "pointcloud":
                    if len(view["observations"]) < self.cfg.min_num_points:
                        continue
                elif self.cfg.filter_for is not None:
                    raise ValueError(f"Unknown filtering: {self.cfg.filter_for}")
                names_select.append((scene, seq, name))
            logger.info(
                "%s: Keep %d/%d images after filtering for %s.",
                stage,
                len(names_select),
                len(names),
                self.cfg.filter_for,
            )
            self.splits[stage] = names_select

    def parse_splits(self, split_arg, names):
        if split_arg is None:
            self.splits = {
                "train": names,
                "val": names,
            }
        elif isinstance(split_arg, int):
            names = np.random.RandomState(self.cfg.seed).permutation(names).tolist()
            self.splits = {
                "train": names[split_arg:],
                "val": names[:split_arg],
            }
        elif isinstance(split_arg, DictConfig):
            scenes_val = set(split_arg.val)
            scenes_train = set(split_arg.train)
            assert len(scenes_val - set(self.cfg.scenes)) == 0
            assert len(scenes_train - set(self.cfg.scenes)) == 0
            self.splits = {
                "train": [n for n in names if n[0] in scenes_train],
                "val": [n for n in names if n[0] in scenes_val],
            }
        elif isinstance(split_arg, str):
            if (self.root / split_arg).exists():
                # Common split file.
                with (self.root / split_arg).open("r") as fp:
                    splits = json.load(fp)
            else:
                # Per-scene split file.
                splits = defaultdict(dict)
                for scene in self.cfg.scenes:
                    with (self.root / split_arg.format(scene=scene)).open("r") as fp:
                        scene_splits = json.load(fp)
                    for split_name in scene_splits:
                        splits[split_name][scene] = scene_splits[split_name]
            splits = {
                split_name: {scene: set(ids) for scene, ids in split.items()}
                for split_name, split in splits.items()
            }
            self.splits = {}
            for split_name, split in splits.items():
                self.splits[split_name] = [
                    (scene, *arg, name)
                    for scene, *arg, name in names
                    if scene in split and int(name.rsplit("_", 1)[0]) in split[scene]
                ]
        else:
            raise ValueError(split_arg)

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.splits[stage],
            self.data[stage],
            self.image_dirs,
            self.tile_managers,
            image_ext=".jpg",
        )

    def dataloader(
        self,
        stage: str,
        shuffle: bool = False,
        num_workers: int = None,
        sampler: Optional[torchdata.Sampler] = None,
    ):
        dataset = self.dataset(stage)
        cfg = self.cfg["loading"][stage]
        batch_size = cfg["batch_size"]

        # Validate batch size is a multiple of views per panorama (n)
        # views_per_panorama = self.num_views // 2
        # views_per_pair = 2
        
        if batch_size % self.num_views != 0:
            raise ValueError(
                f"Batch size ({batch_size}) must be a multiple of views per panorama ({self.num_views})."
            )
        
        num_workers = cfg["num_workers"] if num_workers is None else num_workers

        is_distributed = dist.is_available() and dist.is_initialized()
        if stage == 'train' and is_distributed:
            sampler = DistributedGroupSampler(
                dataset,
                samples_per_group = self.num_views,
                shuffle = True,
            )
            shuffle = False # shuffle of Dataloader should be False when sampler is not None
        else:
            shuffle = shuffle

        loader = torchdata.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)

    def sequence_dataset(self, stage: str, **kwargs):
        keys = self.splits[stage]
        seq2indices = defaultdict(list)
        for index, (_, seq, _) in enumerate(keys):
            seq2indices[seq].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(self.data[stage], indices, **kwargs)
            for i, sub_indices in enumerate(chunks):
                chunk2indices[seq, i] = sub_indices
        # store the index of each chunk in its sequence
        chunk_indices = torch.full((len(keys),), -1)
        for (_, chunk_index), idx in chunk2indices.items():
            chunk_indices[idx] = chunk_index
        self.data[stage]["chunk_index"] = chunk_indices
        dataset = self.dataset(stage)
        return dataset, chunk2indices

    def sequence_dataloader(self, stage: str, shuffle: bool = False, **kwargs):
        dataset, chunk2idx = self.sequence_dataset(stage, **kwargs)
        chunk_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(chunk_keys))
            chunk_keys = [chunk_keys[i] for i in perm]
        key_indices = [i for key in chunk_keys for i in chunk2idx[key]]
        num_workers = self.cfg["loading"][stage]["num_workers"]
        loader = torchdata.DataLoader(
            dataset,
            batch_size=None,
            sampler=key_indices,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return loader, chunk_keys, chunk2idx

class MapillaryPanoFoggyDataModule(pl.LightningDataModule):
    dump_filename = "dump.json"
    # images_archive = "images.tar.gz"
    images_dirname = "images/"

    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "mapillary",
        # paths and fetch
        "data_dir": DATASETS_PATH / "MGL_final",
        "noise_data_dir": DATASETS_PATH / "MGL_foggy",
        "local_dir": None,
        "tiles_filename": "tiles.pkl",
        "scenes": "???",
        "split": None,
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 3, "num_workers": 3},
        },
        "filter_for": None,
        "filter_by_ground_angle": None,
        "min_num_points": "???",
    }

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.num_views = 3
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.noise_root = Path(self.cfg.noise_data_dir)
        self.local_dir = self.cfg.local_dir or os.environ.get("TMPDIR")
        if self.local_dir is not None:
            self.local_dir = Path(self.local_dir, "MGL_foggy")
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")

    def prepare_data(self):
        for scene in self.cfg.scenes:
            dump_dir = self.root / scene
            assert (dump_dir / self.dump_filename).exists(), dump_dir
            assert (dump_dir / self.cfg.tiles_filename).exists(), dump_dir
            if self.local_dir is None:
                assert (self.noise_root / scene / self.images_dirname).exists(), dump_dir
                continue
            # Cache the folder of images locally to speed up reading
            local_dir = self.local_dir / scene
            if local_dir.exists():
                shutil.rmtree(local_dir)
            local_dir.mkdir(exist_ok=True, parents=True)
            # images_archive = dump_dir / self.images_archive
            # logger.info("Extracting the image archive %s.", images_archive)
            # with tarfile.open(images_archive) as fp:
                # fp.extractall(local_dir)

    def setup(self, stage: Optional[str] = None):
        self.dumps = {}
        self.tile_managers = {}
        self.image_dirs = {}
        names = []

        for scene in self.cfg.scenes:
            logger.info("Loading scene %s.", scene)
            dump_dir = self.root / scene
            logger.info("Loading map tiles %s.", self.cfg.tiles_filename)
            self.tile_managers[scene] = TileManager.load(
                dump_dir / self.cfg.tiles_filename
            )
            groups = self.tile_managers[scene].groups
            if self.cfg.num_classes:  # check consistency
                if set(groups.keys()) != set(self.cfg.num_classes.keys()):
                    raise ValueError(
                        "Inconsistent groups: "
                        f"{groups.keys()} {self.cfg.num_classes.keys()}"
                    )
                for k in groups:
                    if len(groups[k]) != self.cfg.num_classes[k]:
                        raise ValueError(
                            f"{k}: {len(groups[k])} vs {self.cfg.num_classes[k]}"
                        )
            ppm = self.tile_managers[scene].ppm
            if ppm != self.cfg.pixel_per_meter:
                raise ValueError(
                    "The tile manager and the config/model have different ground "
                    f"resolutions: {ppm} vs {self.cfg.pixel_per_meter}"
                )

            logger.info("Loading dump json file %s.", self.dump_filename)
            with (dump_dir / self.dump_filename).open("r") as fp:
                self.dumps[scene] = pack_dump_dict(json.load(fp))
            for seq, per_seq in self.dumps[scene].items():
                for cam_id, cam_dict in per_seq["cameras"].items():
                    if cam_dict["model"] != "PINHOLE":
                        raise ValueError(
                            "Unsupported camera model: "
                            f"{cam_dict['model']} for {scene},{seq},{cam_id}"
                        )

            self.image_dirs[scene] = (
                # (self.local_dir or self.root) / scene / self.images_dirname
                self.noise_root / scene / self.images_dirname
            )
            assert self.image_dirs[scene].exists(), self.image_dirs[scene]

            for seq, data in self.dumps[scene].items():
                for name in data["views"]:
                    names.append((scene, seq, name))

        self.parse_splits(self.cfg.split, names)
        if self.cfg.filter_for is not None:
            self.filter_elements()
        self.pack_data()

    def pack_data(self):
        # We pack the data into compact tensors
        # that can be shared across processes without copy.
        # We group views by panorama to enable panorama-based batching.
        exclude = {
            "compass_angle",
            "compass_accuracy",
            "gps_accuracy",
            "chunk_key",
            "panorama_offset",
        }
        cameras = {
            scene: {seq: per_seq["cameras"] for seq, per_seq in per_scene.items()}
            for scene, per_scene in self.dumps.items()
        }
        points = {
            scene: {
                seq: {
                    i: torch.from_numpy(p) for i, p in per_seq.get("points", {}).items()
                }
                for seq, per_seq in per_scene.items()
            }
            for scene, per_scene in self.dumps.items()
        }
        self.data = {}
        self.panorama_groups = {}  # Store panorama grouping info

        for stage, names in self.splits.items():
            # Group views by panorama ID
            panorama_views = self._group_views_by_panorama(names)

            # Validate that we have complete panoramas (n views each)
            incomplete_panoramas = [
                p_id for p_id, views in panorama_views.items() if len(views) != self.num_views
            ]
            if incomplete_panoramas:
                logger.warning(
                    f"Found {len(incomplete_panoramas)} incomplete panoramas "
                    f"in {stage} set. These will be excluded. "
                    f"First few: {incomplete_panoramas[:5]}"
                )
                # Keep only complete panoramas
                panorama_views = {
                    p_id: views
                    for p_id, views in panorama_views.items()
                    if len(views) == self.num_views
                }

            # Store panorama groups for this stage
            self.panorama_groups[stage] = panorama_views

            # Create flattened names list maintaining panorama view order
            ordered_names = []
            for p_id in sorted(panorama_views.keys()):
                views = panorama_views[p_id]
                # Sort views by suffix for consistent order:
                # suffix_order = {"front": 0, "left": 1, "back": 2, "right": 3}
                # suffix_order = {"view1": 0, "view2": 1, "view3": 2}
                suffix_order = {"view1": 0, "view2": 1, "view3": 2}
                # suffix_order = {"v1": 0, "v2": 1, "v3": 2, "v4": 3, "v5": 4, "v6": 5, "v7": 6, "v8": 7, "v9": 8}

                views_sorted = sorted(
                    views, key=lambda x: suffix_order.get(x[2].split("_")[-1], 999)
                )
                ordered_names.extend(views_sorted)

            # Pack data as before but with grouped panorama ordering
            if ordered_names:
                view = self.dumps[ordered_names[0][0]][ordered_names[0][1]]["views"][
                    ordered_names[0][2]
                ]
                data = {k: [] for k in view.keys() - exclude}
                for scene, seq, name in ordered_names:
                    for k in data:
                        data[k].append(
                            self.dumps[scene][seq]["views"][name].get(k, None)
                        )
                for k in data:
                    v = np.array(data[k])
                    if np.issubdtype(v.dtype, np.integer) or np.issubdtype(
                        v.dtype, np.floating
                    ):
                        v = torch.from_numpy(v)
                    data[k] = v
                data["cameras"] = cameras
                data["points"] = points
                self.data[stage] = data
                self.splits[stage] = np.array(ordered_names)
            else:
                # Empty stage
                self.data[stage] = {"cameras": cameras, "points": points}
                self.splits[stage] = np.array([])

    def _group_views_by_panorama(self, names):
        """Group view names by panorama ID.

        Args:
            names: List of (scene, seq, name) tuples

        Returns:
            Dict mapping panorama_id to list of (scene, seq, name) tuples
        """
        panorama_views = defaultdict(list)
        for scene, seq, name in names:
            # Extract panorama ID from view name (everything before the last underscore)
            if "_" in name:
                panorama_id = name.rsplit("_", 1)[0]
                panorama_views[panorama_id].append((scene, seq, name))
            else:
                # Handle non-panoramic views if any exist
                panorama_views[name].append((scene, seq, name))
        return dict(panorama_views)

    def filter_elements(self):
        for stage, names in self.splits.items():
            names_select = []
            for scene, seq, name in names:
                view = self.dumps[scene][seq]["views"][name]
                if self.cfg.filter_for == "ground_plane":
                    if not (1.0 <= view["height"] <= 3.0):
                        continue
                    planes = self.dumps[scene][seq].get("plane")
                    if planes is not None:
                        inliers = planes[str(view["chunk_id"])][-1]
                        if inliers < 10:
                            continue
                    if self.cfg.filter_by_ground_angle is not None:
                        plane = np.array(view["plane_params"])
                        normal = plane[:3] / np.linalg.norm(plane[:3])
                        angle = np.rad2deg(np.arccos(np.abs(normal[-1])))
                        if angle > self.cfg.filter_by_ground_angle:
                            continue
                elif self.cfg.filter_for == "pointcloud":
                    if len(view["observations"]) < self.cfg.min_num_points:
                        continue
                elif self.cfg.filter_for is not None:
                    raise ValueError(f"Unknown filtering: {self.cfg.filter_for}")
                names_select.append((scene, seq, name))
            logger.info(
                "%s: Keep %d/%d images after filtering for %s.",
                stage,
                len(names_select),
                len(names),
                self.cfg.filter_for,
            )
            self.splits[stage] = names_select

    def parse_splits(self, split_arg, names):
        if split_arg is None:
            self.splits = {
                "train": names,
                "val": names,
            }
        elif isinstance(split_arg, int):
            names = np.random.RandomState(self.cfg.seed).permutation(names).tolist()
            self.splits = {
                "train": names[split_arg:],
                "val": names[:split_arg],
            }
        elif isinstance(split_arg, DictConfig):
            scenes_val = set(split_arg.val)
            scenes_train = set(split_arg.train)
            assert len(scenes_val - set(self.cfg.scenes)) == 0
            assert len(scenes_train - set(self.cfg.scenes)) == 0
            self.splits = {
                "train": [n for n in names if n[0] in scenes_train],
                "val": [n for n in names if n[0] in scenes_val],
            }
        elif isinstance(split_arg, str):
            if (self.root / split_arg).exists():
                # Common split file.
                with (self.root / split_arg).open("r") as fp:
                    splits = json.load(fp)
            else:
                # Per-scene split file.
                splits = defaultdict(dict)
                for scene in self.cfg.scenes:
                    with (self.root / split_arg.format(scene=scene)).open("r") as fp:
                        scene_splits = json.load(fp)
                    for split_name in scene_splits:
                        splits[split_name][scene] = scene_splits[split_name]
            splits = {
                split_name: {scene: set(ids) for scene, ids in split.items()}
                for split_name, split in splits.items()
            }
            self.splits = {}
            for split_name, split in splits.items():
                self.splits[split_name] = [
                    (scene, *arg, name)
                    for scene, *arg, name in names
                    if scene in split and int(name.rsplit("_", 1)[0]) in split[scene]
                ]
        else:
            raise ValueError(split_arg)

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.splits[stage],
            self.data[stage],
            self.image_dirs,
            self.tile_managers,
            image_ext=".jpg",
        )

    def dataloader(
        self,
        stage: str,
        shuffle: bool = False,
        num_workers: int = None,
        sampler: Optional[torchdata.Sampler] = None,
    ):
        dataset = self.dataset(stage)
        cfg = self.cfg["loading"][stage]
        batch_size = cfg["batch_size"]

        # Validate batch size is a multiple of views per panorama (n)
        # views_per_panorama = self.num_views // 2
        # views_per_pair = 2
        
        if batch_size % self.num_views != 0:
            raise ValueError(
                f"Batch size ({batch_size}) must be a multiple of views per panorama ({self.num_views})."
            )
        
        num_workers = cfg["num_workers"] if num_workers is None else num_workers

        is_distributed = dist.is_available() and dist.is_initialized()
        if stage == 'train' and is_distributed:
            sampler = DistributedGroupSampler(
                dataset,
                samples_per_group = self.num_views,
                shuffle = True,
            )
            shuffle = False # shuffle of Dataloader should be False when sampler is not None
        else:
            shuffle = shuffle

        loader = torchdata.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)

    def sequence_dataset(self, stage: str, **kwargs):
        keys = self.splits[stage]
        seq2indices = defaultdict(list)
        for index, (_, seq, _) in enumerate(keys):
            seq2indices[seq].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(self.data[stage], indices, **kwargs)
            for i, sub_indices in enumerate(chunks):
                chunk2indices[seq, i] = sub_indices
        # store the index of each chunk in its sequence
        chunk_indices = torch.full((len(keys),), -1)
        for (_, chunk_index), idx in chunk2indices.items():
            chunk_indices[idx] = chunk_index
        self.data[stage]["chunk_index"] = chunk_indices
        dataset = self.dataset(stage)
        return dataset, chunk2indices

    def sequence_dataloader(self, stage: str, shuffle: bool = False, **kwargs):
        dataset, chunk2idx = self.sequence_dataset(stage, **kwargs)
        chunk_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(chunk_keys))
            chunk_keys = [chunk_keys[i] for i in perm]
        key_indices = [i for key in chunk_keys for i in chunk2idx[key]]
        num_workers = self.cfg["loading"][stage]["num_workers"]
        loader = torchdata.DataLoader(
            dataset,
            batch_size=None,
            sampler=key_indices,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return loader, chunk_keys, chunk2idx

class MapillaryPanoSnowyDataModule(pl.LightningDataModule):
    dump_filename = "dump.json"
    # images_archive = "images.tar.gz"
    images_dirname = "images/"

    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "mapillary",
        # paths and fetch
        "data_dir": DATASETS_PATH / "MGL_final",
        "noise_data_dir": DATASETS_PATH / "MGL_snowy",
        "local_dir": None,
        "tiles_filename": "tiles.pkl",
        "scenes": "???",
        "split": None,
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 3, "num_workers": 3},
        },
        "filter_for": None,
        "filter_by_ground_angle": None,
        "min_num_points": "???",
    }

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.num_views = 3
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.noise_root = Path(self.cfg.noise_data_dir)
        self.local_dir = self.cfg.local_dir or os.environ.get("TMPDIR")
        if self.local_dir is not None:
            self.local_dir = Path(self.local_dir, "MGL_snowy")
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")

    def prepare_data(self):
        for scene in self.cfg.scenes:
            dump_dir = self.root / scene
            assert (dump_dir / self.dump_filename).exists(), dump_dir
            assert (dump_dir / self.cfg.tiles_filename).exists(), dump_dir
            if self.local_dir is None:
                assert (self.noise_root / scene / self.images_dirname).exists(), dump_dir
                continue
            # Cache the folder of images locally to speed up reading
            local_dir = self.local_dir / scene
            if local_dir.exists():
                shutil.rmtree(local_dir)
            local_dir.mkdir(exist_ok=True, parents=True)
            # images_archive = dump_dir / self.images_archive
            # logger.info("Extracting the image archive %s.", images_archive)
            # with tarfile.open(images_archive) as fp:
                # fp.extractall(local_dir)

    def setup(self, stage: Optional[str] = None):
        self.dumps = {}
        self.tile_managers = {}
        self.image_dirs = {}
        names = []

        for scene in self.cfg.scenes:
            logger.info("Loading scene %s.", scene)
            dump_dir = self.root / scene
            logger.info("Loading map tiles %s.", self.cfg.tiles_filename)
            self.tile_managers[scene] = TileManager.load(
                dump_dir / self.cfg.tiles_filename
            )
            groups = self.tile_managers[scene].groups
            if self.cfg.num_classes:  # check consistency
                if set(groups.keys()) != set(self.cfg.num_classes.keys()):
                    raise ValueError(
                        "Inconsistent groups: "
                        f"{groups.keys()} {self.cfg.num_classes.keys()}"
                    )
                for k in groups:
                    if len(groups[k]) != self.cfg.num_classes[k]:
                        raise ValueError(
                            f"{k}: {len(groups[k])} vs {self.cfg.num_classes[k]}"
                        )
            ppm = self.tile_managers[scene].ppm
            if ppm != self.cfg.pixel_per_meter:
                raise ValueError(
                    "The tile manager and the config/model have different ground "
                    f"resolutions: {ppm} vs {self.cfg.pixel_per_meter}"
                )

            logger.info("Loading dump json file %s.", self.dump_filename)
            with (dump_dir / self.dump_filename).open("r") as fp:
                self.dumps[scene] = pack_dump_dict(json.load(fp))
            for seq, per_seq in self.dumps[scene].items():
                for cam_id, cam_dict in per_seq["cameras"].items():
                    if cam_dict["model"] != "PINHOLE":
                        raise ValueError(
                            "Unsupported camera model: "
                            f"{cam_dict['model']} for {scene},{seq},{cam_id}"
                        )

            self.image_dirs[scene] = (
                # (self.local_dir or self.root) / scene / self.images_dirname
                self.noise_root / scene / self.images_dirname
            )
            assert self.image_dirs[scene].exists(), self.image_dirs[scene]

            for seq, data in self.dumps[scene].items():
                for name in data["views"]:
                    names.append((scene, seq, name))

        self.parse_splits(self.cfg.split, names)
        if self.cfg.filter_for is not None:
            self.filter_elements()
        self.pack_data()

    def pack_data(self):
        # We pack the data into compact tensors
        # that can be shared across processes without copy.
        # We group views by panorama to enable panorama-based batching.
        exclude = {
            "compass_angle",
            "compass_accuracy",
            "gps_accuracy",
            "chunk_key",
            "panorama_offset",
        }
        cameras = {
            scene: {seq: per_seq["cameras"] for seq, per_seq in per_scene.items()}
            for scene, per_scene in self.dumps.items()
        }
        points = {
            scene: {
                seq: {
                    i: torch.from_numpy(p) for i, p in per_seq.get("points", {}).items()
                }
                for seq, per_seq in per_scene.items()
            }
            for scene, per_scene in self.dumps.items()
        }
        self.data = {}
        self.panorama_groups = {}  # Store panorama grouping info

        for stage, names in self.splits.items():
            # Group views by panorama ID
            panorama_views = self._group_views_by_panorama(names)

            # Validate that we have complete panoramas (n views each)
            incomplete_panoramas = [
                p_id for p_id, views in panorama_views.items() if len(views) != self.num_views
            ]
            if incomplete_panoramas:
                logger.warning(
                    f"Found {len(incomplete_panoramas)} incomplete panoramas "
                    f"in {stage} set. These will be excluded. "
                    f"First few: {incomplete_panoramas[:5]}"
                )
                # Keep only complete panoramas
                panorama_views = {
                    p_id: views
                    for p_id, views in panorama_views.items()
                    if len(views) == self.num_views
                }

            # Store panorama groups for this stage
            self.panorama_groups[stage] = panorama_views

            # Create flattened names list maintaining panorama view order
            ordered_names = []
            for p_id in sorted(panorama_views.keys()):
                views = panorama_views[p_id]
                # Sort views by suffix for consistent order:
                # suffix_order = {"front": 0, "left": 1, "back": 2, "right": 3}
                # suffix_order = {"view1": 0, "view2": 1, "view3": 2}
                suffix_order = {"view1": 0, "view2": 1, "view3": 2}
                # suffix_order = {"v1": 0, "v2": 1, "v3": 2, "v4": 3, "v5": 4, "v6": 5, "v7": 6, "v8": 7, "v9": 8}

                views_sorted = sorted(
                    views, key=lambda x: suffix_order.get(x[2].split("_")[-1], 999)
                )
                ordered_names.extend(views_sorted)

            # Pack data as before but with grouped panorama ordering
            if ordered_names:
                view = self.dumps[ordered_names[0][0]][ordered_names[0][1]]["views"][
                    ordered_names[0][2]
                ]
                data = {k: [] for k in view.keys() - exclude}
                for scene, seq, name in ordered_names:
                    for k in data:
                        data[k].append(
                            self.dumps[scene][seq]["views"][name].get(k, None)
                        )
                for k in data:
                    v = np.array(data[k])
                    if np.issubdtype(v.dtype, np.integer) or np.issubdtype(
                        v.dtype, np.floating
                    ):
                        v = torch.from_numpy(v)
                    data[k] = v
                data["cameras"] = cameras
                data["points"] = points
                self.data[stage] = data
                self.splits[stage] = np.array(ordered_names)
            else:
                # Empty stage
                self.data[stage] = {"cameras": cameras, "points": points}
                self.splits[stage] = np.array([])

    def _group_views_by_panorama(self, names):
        """Group view names by panorama ID.

        Args:
            names: List of (scene, seq, name) tuples

        Returns:
            Dict mapping panorama_id to list of (scene, seq, name) tuples
        """
        panorama_views = defaultdict(list)
        for scene, seq, name in names:
            # Extract panorama ID from view name (everything before the last underscore)
            if "_" in name:
                panorama_id = name.rsplit("_", 1)[0]
                panorama_views[panorama_id].append((scene, seq, name))
            else:
                # Handle non-panoramic views if any exist
                panorama_views[name].append((scene, seq, name))
        return dict(panorama_views)

    def filter_elements(self):
        for stage, names in self.splits.items():
            names_select = []
            for scene, seq, name in names:
                view = self.dumps[scene][seq]["views"][name]
                if self.cfg.filter_for == "ground_plane":
                    if not (1.0 <= view["height"] <= 3.0):
                        continue
                    planes = self.dumps[scene][seq].get("plane")
                    if planes is not None:
                        inliers = planes[str(view["chunk_id"])][-1]
                        if inliers < 10:
                            continue
                    if self.cfg.filter_by_ground_angle is not None:
                        plane = np.array(view["plane_params"])
                        normal = plane[:3] / np.linalg.norm(plane[:3])
                        angle = np.rad2deg(np.arccos(np.abs(normal[-1])))
                        if angle > self.cfg.filter_by_ground_angle:
                            continue
                elif self.cfg.filter_for == "pointcloud":
                    if len(view["observations"]) < self.cfg.min_num_points:
                        continue
                elif self.cfg.filter_for is not None:
                    raise ValueError(f"Unknown filtering: {self.cfg.filter_for}")
                names_select.append((scene, seq, name))
            logger.info(
                "%s: Keep %d/%d images after filtering for %s.",
                stage,
                len(names_select),
                len(names),
                self.cfg.filter_for,
            )
            self.splits[stage] = names_select

    def parse_splits(self, split_arg, names):
        if split_arg is None:
            self.splits = {
                "train": names,
                "val": names,
            }
        elif isinstance(split_arg, int):
            names = np.random.RandomState(self.cfg.seed).permutation(names).tolist()
            self.splits = {
                "train": names[split_arg:],
                "val": names[:split_arg],
            }
        elif isinstance(split_arg, DictConfig):
            scenes_val = set(split_arg.val)
            scenes_train = set(split_arg.train)
            assert len(scenes_val - set(self.cfg.scenes)) == 0
            assert len(scenes_train - set(self.cfg.scenes)) == 0
            self.splits = {
                "train": [n for n in names if n[0] in scenes_train],
                "val": [n for n in names if n[0] in scenes_val],
            }
        elif isinstance(split_arg, str):
            if (self.root / split_arg).exists():
                # Common split file.
                with (self.root / split_arg).open("r") as fp:
                    splits = json.load(fp)
            else:
                # Per-scene split file.
                splits = defaultdict(dict)
                for scene in self.cfg.scenes:
                    with (self.root / split_arg.format(scene=scene)).open("r") as fp:
                        scene_splits = json.load(fp)
                    for split_name in scene_splits:
                        splits[split_name][scene] = scene_splits[split_name]
            splits = {
                split_name: {scene: set(ids) for scene, ids in split.items()}
                for split_name, split in splits.items()
            }
            self.splits = {}
            for split_name, split in splits.items():
                self.splits[split_name] = [
                    (scene, *arg, name)
                    for scene, *arg, name in names
                    if scene in split and int(name.rsplit("_", 1)[0]) in split[scene]
                ]
        else:
            raise ValueError(split_arg)

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.splits[stage],
            self.data[stage],
            self.image_dirs,
            self.tile_managers,
            image_ext=".jpg",
        )

    def dataloader(
        self,
        stage: str,
        shuffle: bool = False,
        num_workers: int = None,
        sampler: Optional[torchdata.Sampler] = None,
    ):
        dataset = self.dataset(stage)
        cfg = self.cfg["loading"][stage]
        batch_size = cfg["batch_size"]

        # Validate batch size is a multiple of views per panorama (n)
        # views_per_panorama = self.num_views // 2
        # views_per_pair = 2
        
        if batch_size % self.num_views != 0:
            raise ValueError(
                f"Batch size ({batch_size}) must be a multiple of views per panorama ({self.num_views})."
            )
        
        num_workers = cfg["num_workers"] if num_workers is None else num_workers

        is_distributed = dist.is_available() and dist.is_initialized()
        if stage == 'train' and is_distributed:
            sampler = DistributedGroupSampler(
                dataset,
                samples_per_group = self.num_views,
                shuffle = True,
            )
            shuffle = False # shuffle of Dataloader should be False when sampler is not None
        else:
            shuffle = shuffle

        loader = torchdata.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)

    def sequence_dataset(self, stage: str, **kwargs):
        keys = self.splits[stage]
        seq2indices = defaultdict(list)
        for index, (_, seq, _) in enumerate(keys):
            seq2indices[seq].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(self.data[stage], indices, **kwargs)
            for i, sub_indices in enumerate(chunks):
                chunk2indices[seq, i] = sub_indices
        # store the index of each chunk in its sequence
        chunk_indices = torch.full((len(keys),), -1)
        for (_, chunk_index), idx in chunk2indices.items():
            chunk_indices[idx] = chunk_index
        self.data[stage]["chunk_index"] = chunk_indices
        dataset = self.dataset(stage)
        return dataset, chunk2indices

    def sequence_dataloader(self, stage: str, shuffle: bool = False, **kwargs):
        dataset, chunk2idx = self.sequence_dataset(stage, **kwargs)
        chunk_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(chunk_keys))
            chunk_keys = [chunk_keys[i] for i in perm]
        key_indices = [i for key in chunk_keys for i in chunk2idx[key]]
        num_workers = self.cfg["loading"][stage]["num_workers"]
        loader = torchdata.DataLoader(
            dataset,
            batch_size=None,
            sampler=key_indices,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return loader, chunk_keys, chunk2idx

class MapillaryPanoOverExposureDataModule(pl.LightningDataModule):
    dump_filename = "dump.json"
    # images_archive = "images.tar.gz"
    images_dirname = "images/"

    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "mapillary",
        # paths and fetch
        "data_dir": DATASETS_PATH / "MGL_final",
        "noise_data_dir": DATASETS_PATH / "MGL_over_exposure",
        "local_dir": None,
        "tiles_filename": "tiles.pkl",
        "scenes": "???",
        "split": None,
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 3, "num_workers": 3},
        },
        "filter_for": None,
        "filter_by_ground_angle": None,
        "min_num_points": "???",
    }

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.num_views = 3
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.noise_root = Path(self.cfg.noise_data_dir)
        self.local_dir = self.cfg.local_dir or os.environ.get("TMPDIR")
        if self.local_dir is not None:
            self.local_dir = Path(self.local_dir, "MGL_over_exposure")
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")

    def prepare_data(self):
        for scene in self.cfg.scenes:
            dump_dir = self.root / scene
            assert (dump_dir / self.dump_filename).exists(), dump_dir
            assert (dump_dir / self.cfg.tiles_filename).exists(), dump_dir
            if self.local_dir is None:
                assert (self.noise_root / scene / self.images_dirname).exists(), dump_dir
                continue
            # Cache the folder of images locally to speed up reading
            local_dir = self.local_dir / scene
            if local_dir.exists():
                shutil.rmtree(local_dir)
            local_dir.mkdir(exist_ok=True, parents=True)
            # images_archive = dump_dir / self.images_archive
            # logger.info("Extracting the image archive %s.", images_archive)
            # with tarfile.open(images_archive) as fp:
                # fp.extractall(local_dir)

    def setup(self, stage: Optional[str] = None):
        self.dumps = {}
        self.tile_managers = {}
        self.image_dirs = {}
        names = []

        for scene in self.cfg.scenes:
            logger.info("Loading scene %s.", scene)
            dump_dir = self.root / scene
            logger.info("Loading map tiles %s.", self.cfg.tiles_filename)
            self.tile_managers[scene] = TileManager.load(
                dump_dir / self.cfg.tiles_filename
            )
            groups = self.tile_managers[scene].groups
            if self.cfg.num_classes:  # check consistency
                if set(groups.keys()) != set(self.cfg.num_classes.keys()):
                    raise ValueError(
                        "Inconsistent groups: "
                        f"{groups.keys()} {self.cfg.num_classes.keys()}"
                    )
                for k in groups:
                    if len(groups[k]) != self.cfg.num_classes[k]:
                        raise ValueError(
                            f"{k}: {len(groups[k])} vs {self.cfg.num_classes[k]}"
                        )
            ppm = self.tile_managers[scene].ppm
            if ppm != self.cfg.pixel_per_meter:
                raise ValueError(
                    "The tile manager and the config/model have different ground "
                    f"resolutions: {ppm} vs {self.cfg.pixel_per_meter}"
                )

            logger.info("Loading dump json file %s.", self.dump_filename)
            with (dump_dir / self.dump_filename).open("r") as fp:
                self.dumps[scene] = pack_dump_dict(json.load(fp))
            for seq, per_seq in self.dumps[scene].items():
                for cam_id, cam_dict in per_seq["cameras"].items():
                    if cam_dict["model"] != "PINHOLE":
                        raise ValueError(
                            "Unsupported camera model: "
                            f"{cam_dict['model']} for {scene},{seq},{cam_id}"
                        )

            self.image_dirs[scene] = (
                # (self.local_dir or self.root) / scene / self.images_dirname
                self.noise_root / scene / self.images_dirname
            )
            assert self.image_dirs[scene].exists(), self.image_dirs[scene]

            for seq, data in self.dumps[scene].items():
                for name in data["views"]:
                    names.append((scene, seq, name))

        self.parse_splits(self.cfg.split, names)
        if self.cfg.filter_for is not None:
            self.filter_elements()
        self.pack_data()

    def pack_data(self):
        # We pack the data into compact tensors
        # that can be shared across processes without copy.
        # We group views by panorama to enable panorama-based batching.
        exclude = {
            "compass_angle",
            "compass_accuracy",
            "gps_accuracy",
            "chunk_key",
            "panorama_offset",
        }
        cameras = {
            scene: {seq: per_seq["cameras"] for seq, per_seq in per_scene.items()}
            for scene, per_scene in self.dumps.items()
        }
        points = {
            scene: {
                seq: {
                    i: torch.from_numpy(p) for i, p in per_seq.get("points", {}).items()
                }
                for seq, per_seq in per_scene.items()
            }
            for scene, per_scene in self.dumps.items()
        }
        self.data = {}
        self.panorama_groups = {}  # Store panorama grouping info

        for stage, names in self.splits.items():
            # Group views by panorama ID
            panorama_views = self._group_views_by_panorama(names)

            # Validate that we have complete panoramas (n views each)
            incomplete_panoramas = [
                p_id for p_id, views in panorama_views.items() if len(views) != self.num_views
            ]
            if incomplete_panoramas:
                logger.warning(
                    f"Found {len(incomplete_panoramas)} incomplete panoramas "
                    f"in {stage} set. These will be excluded. "
                    f"First few: {incomplete_panoramas[:5]}"
                )
                # Keep only complete panoramas
                panorama_views = {
                    p_id: views
                    for p_id, views in panorama_views.items()
                    if len(views) == self.num_views
                }

            # Store panorama groups for this stage
            self.panorama_groups[stage] = panorama_views

            # Create flattened names list maintaining panorama view order
            ordered_names = []
            for p_id in sorted(panorama_views.keys()):
                views = panorama_views[p_id]
                # Sort views by suffix for consistent order:
                # suffix_order = {"front": 0, "left": 1, "back": 2, "right": 3}
                # suffix_order = {"view1": 0, "view2": 1, "view3": 2}
                suffix_order = {"view1": 0, "view2": 1, "view3": 2}
                # suffix_order = {"v1": 0, "v2": 1, "v3": 2, "v4": 3, "v5": 4, "v6": 5, "v7": 6, "v8": 7, "v9": 8}

                views_sorted = sorted(
                    views, key=lambda x: suffix_order.get(x[2].split("_")[-1], 999)
                )
                ordered_names.extend(views_sorted)

            # Pack data as before but with grouped panorama ordering
            if ordered_names:
                view = self.dumps[ordered_names[0][0]][ordered_names[0][1]]["views"][
                    ordered_names[0][2]
                ]
                data = {k: [] for k in view.keys() - exclude}
                for scene, seq, name in ordered_names:
                    for k in data:
                        data[k].append(
                            self.dumps[scene][seq]["views"][name].get(k, None)
                        )
                for k in data:
                    v = np.array(data[k])
                    if np.issubdtype(v.dtype, np.integer) or np.issubdtype(
                        v.dtype, np.floating
                    ):
                        v = torch.from_numpy(v)
                    data[k] = v
                data["cameras"] = cameras
                data["points"] = points
                self.data[stage] = data
                self.splits[stage] = np.array(ordered_names)
            else:
                # Empty stage
                self.data[stage] = {"cameras": cameras, "points": points}
                self.splits[stage] = np.array([])

    def _group_views_by_panorama(self, names):
        """Group view names by panorama ID.

        Args:
            names: List of (scene, seq, name) tuples

        Returns:
            Dict mapping panorama_id to list of (scene, seq, name) tuples
        """
        panorama_views = defaultdict(list)
        for scene, seq, name in names:
            # Extract panorama ID from view name (everything before the last underscore)
            if "_" in name:
                panorama_id = name.rsplit("_", 1)[0]
                panorama_views[panorama_id].append((scene, seq, name))
            else:
                # Handle non-panoramic views if any exist
                panorama_views[name].append((scene, seq, name))
        return dict(panorama_views)

    def filter_elements(self):
        for stage, names in self.splits.items():
            names_select = []
            for scene, seq, name in names:
                view = self.dumps[scene][seq]["views"][name]
                if self.cfg.filter_for == "ground_plane":
                    if not (1.0 <= view["height"] <= 3.0):
                        continue
                    planes = self.dumps[scene][seq].get("plane")
                    if planes is not None:
                        inliers = planes[str(view["chunk_id"])][-1]
                        if inliers < 10:
                            continue
                    if self.cfg.filter_by_ground_angle is not None:
                        plane = np.array(view["plane_params"])
                        normal = plane[:3] / np.linalg.norm(plane[:3])
                        angle = np.rad2deg(np.arccos(np.abs(normal[-1])))
                        if angle > self.cfg.filter_by_ground_angle:
                            continue
                elif self.cfg.filter_for == "pointcloud":
                    if len(view["observations"]) < self.cfg.min_num_points:
                        continue
                elif self.cfg.filter_for is not None:
                    raise ValueError(f"Unknown filtering: {self.cfg.filter_for}")
                names_select.append((scene, seq, name))
            logger.info(
                "%s: Keep %d/%d images after filtering for %s.",
                stage,
                len(names_select),
                len(names),
                self.cfg.filter_for,
            )
            self.splits[stage] = names_select

    def parse_splits(self, split_arg, names):
        if split_arg is None:
            self.splits = {
                "train": names,
                "val": names,
            }
        elif isinstance(split_arg, int):
            names = np.random.RandomState(self.cfg.seed).permutation(names).tolist()
            self.splits = {
                "train": names[split_arg:],
                "val": names[:split_arg],
            }
        elif isinstance(split_arg, DictConfig):
            scenes_val = set(split_arg.val)
            scenes_train = set(split_arg.train)
            assert len(scenes_val - set(self.cfg.scenes)) == 0
            assert len(scenes_train - set(self.cfg.scenes)) == 0
            self.splits = {
                "train": [n for n in names if n[0] in scenes_train],
                "val": [n for n in names if n[0] in scenes_val],
            }
        elif isinstance(split_arg, str):
            if (self.root / split_arg).exists():
                # Common split file.
                with (self.root / split_arg).open("r") as fp:
                    splits = json.load(fp)
            else:
                # Per-scene split file.
                splits = defaultdict(dict)
                for scene in self.cfg.scenes:
                    with (self.root / split_arg.format(scene=scene)).open("r") as fp:
                        scene_splits = json.load(fp)
                    for split_name in scene_splits:
                        splits[split_name][scene] = scene_splits[split_name]
            splits = {
                split_name: {scene: set(ids) for scene, ids in split.items()}
                for split_name, split in splits.items()
            }
            self.splits = {}
            for split_name, split in splits.items():
                self.splits[split_name] = [
                    (scene, *arg, name)
                    for scene, *arg, name in names
                    if scene in split and int(name.rsplit("_", 1)[0]) in split[scene]
                ]
        else:
            raise ValueError(split_arg)

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.splits[stage],
            self.data[stage],
            self.image_dirs,
            self.tile_managers,
            image_ext=".jpg",
        )

    def dataloader(
        self,
        stage: str,
        shuffle: bool = False,
        num_workers: int = None,
        sampler: Optional[torchdata.Sampler] = None,
    ):
        dataset = self.dataset(stage)
        cfg = self.cfg["loading"][stage]
        batch_size = cfg["batch_size"]

        # Validate batch size is a multiple of views per panorama (n)
        # views_per_panorama = self.num_views // 2
        # views_per_pair = 2
        
        if batch_size % self.num_views != 0:
            raise ValueError(
                f"Batch size ({batch_size}) must be a multiple of views per panorama ({self.num_views})."
            )
        
        num_workers = cfg["num_workers"] if num_workers is None else num_workers

        is_distributed = dist.is_available() and dist.is_initialized()
        if stage == 'train' and is_distributed:
            sampler = DistributedGroupSampler(
                dataset,
                samples_per_group = self.num_views,
                shuffle = True,
            )
            shuffle = False # shuffle of Dataloader should be False when sampler is not None
        else:
            shuffle = shuffle

        loader = torchdata.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)

    def sequence_dataset(self, stage: str, **kwargs):
        keys = self.splits[stage]
        seq2indices = defaultdict(list)
        for index, (_, seq, _) in enumerate(keys):
            seq2indices[seq].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(self.data[stage], indices, **kwargs)
            for i, sub_indices in enumerate(chunks):
                chunk2indices[seq, i] = sub_indices
        # store the index of each chunk in its sequence
        chunk_indices = torch.full((len(keys),), -1)
        for (_, chunk_index), idx in chunk2indices.items():
            chunk_indices[idx] = chunk_index
        self.data[stage]["chunk_index"] = chunk_indices
        dataset = self.dataset(stage)
        return dataset, chunk2indices

    def sequence_dataloader(self, stage: str, shuffle: bool = False, **kwargs):
        dataset, chunk2idx = self.sequence_dataset(stage, **kwargs)
        chunk_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(chunk_keys))
            chunk_keys = [chunk_keys[i] for i in perm]
        key_indices = [i for key in chunk_keys for i in chunk2idx[key]]
        num_workers = self.cfg["loading"][stage]["num_workers"]
        loader = torchdata.DataLoader(
            dataset,
            batch_size=None,
            sampler=key_indices,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return loader, chunk_keys, chunk2idx

class MapillaryPanoUnderExposureDataModule(pl.LightningDataModule):
    dump_filename = "dump.json"
    # images_archive = "images.tar.gz"
    images_dirname = "images/"

    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "mapillary",
        # paths and fetch
        "data_dir": DATASETS_PATH / "MGL_final",
        "noise_data_dir": DATASETS_PATH / "MGL_under_exposure",
        "local_dir": None,
        "tiles_filename": "tiles.pkl",
        "scenes": "???",
        "split": None,
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 3, "num_workers": 3},
        },
        "filter_for": None,
        "filter_by_ground_angle": None,
        "min_num_points": "???",
    }

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.num_views = 3
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.noise_root = Path(self.cfg.noise_data_dir)
        self.local_dir = self.cfg.local_dir or os.environ.get("TMPDIR")
        if self.local_dir is not None:
            self.local_dir = Path(self.local_dir, "MGL_under_exposure")
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")

    def prepare_data(self):
        for scene in self.cfg.scenes:
            dump_dir = self.root / scene
            assert (dump_dir / self.dump_filename).exists(), dump_dir
            assert (dump_dir / self.cfg.tiles_filename).exists(), dump_dir
            if self.local_dir is None:
                assert (self.noise_root / scene / self.images_dirname).exists(), dump_dir
                continue
            # Cache the folder of images locally to speed up reading
            local_dir = self.local_dir / scene
            if local_dir.exists():
                shutil.rmtree(local_dir)
            local_dir.mkdir(exist_ok=True, parents=True)
            # images_archive = dump_dir / self.images_archive
            # logger.info("Extracting the image archive %s.", images_archive)
            # with tarfile.open(images_archive) as fp:
                # fp.extractall(local_dir)

    def setup(self, stage: Optional[str] = None):
        self.dumps = {}
        self.tile_managers = {}
        self.image_dirs = {}
        names = []

        for scene in self.cfg.scenes:
            logger.info("Loading scene %s.", scene)
            dump_dir = self.root / scene
            logger.info("Loading map tiles %s.", self.cfg.tiles_filename)
            self.tile_managers[scene] = TileManager.load(
                dump_dir / self.cfg.tiles_filename
            )
            groups = self.tile_managers[scene].groups
            if self.cfg.num_classes:  # check consistency
                if set(groups.keys()) != set(self.cfg.num_classes.keys()):
                    raise ValueError(
                        "Inconsistent groups: "
                        f"{groups.keys()} {self.cfg.num_classes.keys()}"
                    )
                for k in groups:
                    if len(groups[k]) != self.cfg.num_classes[k]:
                        raise ValueError(
                            f"{k}: {len(groups[k])} vs {self.cfg.num_classes[k]}"
                        )
            ppm = self.tile_managers[scene].ppm
            if ppm != self.cfg.pixel_per_meter:
                raise ValueError(
                    "The tile manager and the config/model have different ground "
                    f"resolutions: {ppm} vs {self.cfg.pixel_per_meter}"
                )

            logger.info("Loading dump json file %s.", self.dump_filename)
            with (dump_dir / self.dump_filename).open("r") as fp:
                self.dumps[scene] = pack_dump_dict(json.load(fp))
            for seq, per_seq in self.dumps[scene].items():
                for cam_id, cam_dict in per_seq["cameras"].items():
                    if cam_dict["model"] != "PINHOLE":
                        raise ValueError(
                            "Unsupported camera model: "
                            f"{cam_dict['model']} for {scene},{seq},{cam_id}"
                        )

            self.image_dirs[scene] = (
                # (self.local_dir or self.root) / scene / self.images_dirname
                self.noise_root / scene / self.images_dirname
            )
            assert self.image_dirs[scene].exists(), self.image_dirs[scene]

            for seq, data in self.dumps[scene].items():
                for name in data["views"]:
                    names.append((scene, seq, name))

        self.parse_splits(self.cfg.split, names)
        if self.cfg.filter_for is not None:
            self.filter_elements()
        self.pack_data()

    def pack_data(self):
        # We pack the data into compact tensors
        # that can be shared across processes without copy.
        # We group views by panorama to enable panorama-based batching.
        exclude = {
            "compass_angle",
            "compass_accuracy",
            "gps_accuracy",
            "chunk_key",
            "panorama_offset",
        }
        cameras = {
            scene: {seq: per_seq["cameras"] for seq, per_seq in per_scene.items()}
            for scene, per_scene in self.dumps.items()
        }
        points = {
            scene: {
                seq: {
                    i: torch.from_numpy(p) for i, p in per_seq.get("points", {}).items()
                }
                for seq, per_seq in per_scene.items()
            }
            for scene, per_scene in self.dumps.items()
        }
        self.data = {}
        self.panorama_groups = {}  # Store panorama grouping info

        for stage, names in self.splits.items():
            # Group views by panorama ID
            panorama_views = self._group_views_by_panorama(names)

            # Validate that we have complete panoramas (n views each)
            incomplete_panoramas = [
                p_id for p_id, views in panorama_views.items() if len(views) != self.num_views
            ]
            if incomplete_panoramas:
                logger.warning(
                    f"Found {len(incomplete_panoramas)} incomplete panoramas "
                    f"in {stage} set. These will be excluded. "
                    f"First few: {incomplete_panoramas[:5]}"
                )
                # Keep only complete panoramas
                panorama_views = {
                    p_id: views
                    for p_id, views in panorama_views.items()
                    if len(views) == self.num_views
                }

            # Store panorama groups for this stage
            self.panorama_groups[stage] = panorama_views

            # Create flattened names list maintaining panorama view order
            ordered_names = []
            for p_id in sorted(panorama_views.keys()):
                views = panorama_views[p_id]
                # Sort views by suffix for consistent order:
                # suffix_order = {"front": 0, "left": 1, "back": 2, "right": 3}
                # suffix_order = {"view1": 0, "view2": 1, "view3": 2}
                suffix_order = {"view1": 0, "view2": 1, "view3": 2}
                # suffix_order = {"v1": 0, "v2": 1, "v3": 2, "v4": 3, "v5": 4, "v6": 5, "v7": 6, "v8": 7, "v9": 8}

                views_sorted = sorted(
                    views, key=lambda x: suffix_order.get(x[2].split("_")[-1], 999)
                )
                ordered_names.extend(views_sorted)

            # Pack data as before but with grouped panorama ordering
            if ordered_names:
                view = self.dumps[ordered_names[0][0]][ordered_names[0][1]]["views"][
                    ordered_names[0][2]
                ]
                data = {k: [] for k in view.keys() - exclude}
                for scene, seq, name in ordered_names:
                    for k in data:
                        data[k].append(
                            self.dumps[scene][seq]["views"][name].get(k, None)
                        )
                for k in data:
                    v = np.array(data[k])
                    if np.issubdtype(v.dtype, np.integer) or np.issubdtype(
                        v.dtype, np.floating
                    ):
                        v = torch.from_numpy(v)
                    data[k] = v
                data["cameras"] = cameras
                data["points"] = points
                self.data[stage] = data
                self.splits[stage] = np.array(ordered_names)
            else:
                # Empty stage
                self.data[stage] = {"cameras": cameras, "points": points}
                self.splits[stage] = np.array([])

    def _group_views_by_panorama(self, names):
        """Group view names by panorama ID.

        Args:
            names: List of (scene, seq, name) tuples

        Returns:
            Dict mapping panorama_id to list of (scene, seq, name) tuples
        """
        panorama_views = defaultdict(list)
        for scene, seq, name in names:
            # Extract panorama ID from view name (everything before the last underscore)
            if "_" in name:
                panorama_id = name.rsplit("_", 1)[0]
                panorama_views[panorama_id].append((scene, seq, name))
            else:
                # Handle non-panoramic views if any exist
                panorama_views[name].append((scene, seq, name))
        return dict(panorama_views)

    def filter_elements(self):
        for stage, names in self.splits.items():
            names_select = []
            for scene, seq, name in names:
                view = self.dumps[scene][seq]["views"][name]
                if self.cfg.filter_for == "ground_plane":
                    if not (1.0 <= view["height"] <= 3.0):
                        continue
                    planes = self.dumps[scene][seq].get("plane")
                    if planes is not None:
                        inliers = planes[str(view["chunk_id"])][-1]
                        if inliers < 10:
                            continue
                    if self.cfg.filter_by_ground_angle is not None:
                        plane = np.array(view["plane_params"])
                        normal = plane[:3] / np.linalg.norm(plane[:3])
                        angle = np.rad2deg(np.arccos(np.abs(normal[-1])))
                        if angle > self.cfg.filter_by_ground_angle:
                            continue
                elif self.cfg.filter_for == "pointcloud":
                    if len(view["observations"]) < self.cfg.min_num_points:
                        continue
                elif self.cfg.filter_for is not None:
                    raise ValueError(f"Unknown filtering: {self.cfg.filter_for}")
                names_select.append((scene, seq, name))
            logger.info(
                "%s: Keep %d/%d images after filtering for %s.",
                stage,
                len(names_select),
                len(names),
                self.cfg.filter_for,
            )
            self.splits[stage] = names_select

    def parse_splits(self, split_arg, names):
        if split_arg is None:
            self.splits = {
                "train": names,
                "val": names,
            }
        elif isinstance(split_arg, int):
            names = np.random.RandomState(self.cfg.seed).permutation(names).tolist()
            self.splits = {
                "train": names[split_arg:],
                "val": names[:split_arg],
            }
        elif isinstance(split_arg, DictConfig):
            scenes_val = set(split_arg.val)
            scenes_train = set(split_arg.train)
            assert len(scenes_val - set(self.cfg.scenes)) == 0
            assert len(scenes_train - set(self.cfg.scenes)) == 0
            self.splits = {
                "train": [n for n in names if n[0] in scenes_train],
                "val": [n for n in names if n[0] in scenes_val],
            }
        elif isinstance(split_arg, str):
            if (self.root / split_arg).exists():
                # Common split file.
                with (self.root / split_arg).open("r") as fp:
                    splits = json.load(fp)
            else:
                # Per-scene split file.
                splits = defaultdict(dict)
                for scene in self.cfg.scenes:
                    with (self.root / split_arg.format(scene=scene)).open("r") as fp:
                        scene_splits = json.load(fp)
                    for split_name in scene_splits:
                        splits[split_name][scene] = scene_splits[split_name]
            splits = {
                split_name: {scene: set(ids) for scene, ids in split.items()}
                for split_name, split in splits.items()
            }
            self.splits = {}
            for split_name, split in splits.items():
                self.splits[split_name] = [
                    (scene, *arg, name)
                    for scene, *arg, name in names
                    if scene in split and int(name.rsplit("_", 1)[0]) in split[scene]
                ]
        else:
            raise ValueError(split_arg)

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.splits[stage],
            self.data[stage],
            self.image_dirs,
            self.tile_managers,
            image_ext=".jpg",
        )

    def dataloader(
        self,
        stage: str,
        shuffle: bool = False,
        num_workers: int = None,
        sampler: Optional[torchdata.Sampler] = None,
    ):
        dataset = self.dataset(stage)
        cfg = self.cfg["loading"][stage]
        batch_size = cfg["batch_size"]

        # Validate batch size is a multiple of views per panorama (n)
        # views_per_panorama = self.num_views // 2
        # views_per_pair = 2
        
        if batch_size % self.num_views != 0:
            raise ValueError(
                f"Batch size ({batch_size}) must be a multiple of views per panorama ({self.num_views})."
            )
        
        num_workers = cfg["num_workers"] if num_workers is None else num_workers

        is_distributed = dist.is_available() and dist.is_initialized()
        if stage == 'train' and is_distributed:
            sampler = DistributedGroupSampler(
                dataset,
                samples_per_group = self.num_views,
                shuffle = True,
            )
            shuffle = False # shuffle of Dataloader should be False when sampler is not None
        else:
            shuffle = shuffle

        loader = torchdata.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)

    def sequence_dataset(self, stage: str, **kwargs):
        keys = self.splits[stage]
        seq2indices = defaultdict(list)
        for index, (_, seq, _) in enumerate(keys):
            seq2indices[seq].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(self.data[stage], indices, **kwargs)
            for i, sub_indices in enumerate(chunks):
                chunk2indices[seq, i] = sub_indices
        # store the index of each chunk in its sequence
        chunk_indices = torch.full((len(keys),), -1)
        for (_, chunk_index), idx in chunk2indices.items():
            chunk_indices[idx] = chunk_index
        self.data[stage]["chunk_index"] = chunk_indices
        dataset = self.dataset(stage)
        return dataset, chunk2indices

    def sequence_dataloader(self, stage: str, shuffle: bool = False, **kwargs):
        dataset, chunk2idx = self.sequence_dataset(stage, **kwargs)
        chunk_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(chunk_keys))
            chunk_keys = [chunk_keys[i] for i in perm]
        key_indices = [i for key in chunk_keys for i in chunk2idx[key]]
        num_workers = self.cfg["loading"][stage]["num_workers"]
        loader = torchdata.DataLoader(
            dataset,
            batch_size=None,
            sampler=key_indices,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return loader, chunk_keys, chunk2idx

class MapillaryPanoMotionBlurDataModule(pl.LightningDataModule):
    dump_filename = "dump.json"
    # images_archive = "images.tar.gz"
    images_dirname = "images/"

    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "mapillary",
        # paths and fetch
        "data_dir": DATASETS_PATH / "MGL_final",
        "noise_data_dir": DATASETS_PATH / "MGL_motion_blur",
        "local_dir": None,
        "tiles_filename": "tiles.pkl",
        "scenes": "???",
        "split": None,
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 3, "num_workers": 3},
        },
        "filter_for": None,
        "filter_by_ground_angle": None,
        "min_num_points": "???",
    }

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.num_views = 3
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.noise_root = Path(self.cfg.noise_data_dir)
        self.local_dir = self.cfg.local_dir or os.environ.get("TMPDIR")
        if self.local_dir is not None:
            self.local_dir = Path(self.local_dir, "MGL_motion_blur")
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")

    def prepare_data(self):
        for scene in self.cfg.scenes:
            dump_dir = self.root / scene
            assert (dump_dir / self.dump_filename).exists(), dump_dir
            assert (dump_dir / self.cfg.tiles_filename).exists(), dump_dir
            if self.local_dir is None:
                assert (self.noise_root / scene / self.images_dirname).exists(), dump_dir
                continue
            # Cache the folder of images locally to speed up reading
            local_dir = self.local_dir / scene
            if local_dir.exists():
                shutil.rmtree(local_dir)
            local_dir.mkdir(exist_ok=True, parents=True)
            # images_archive = dump_dir / self.images_archive
            # logger.info("Extracting the image archive %s.", images_archive)
            # with tarfile.open(images_archive) as fp:
                # fp.extractall(local_dir)

    def setup(self, stage: Optional[str] = None):
        self.dumps = {}
        self.tile_managers = {}
        self.image_dirs = {}
        names = []

        for scene in self.cfg.scenes:
            logger.info("Loading scene %s.", scene)
            dump_dir = self.root / scene
            logger.info("Loading map tiles %s.", self.cfg.tiles_filename)
            self.tile_managers[scene] = TileManager.load(
                dump_dir / self.cfg.tiles_filename
            )
            groups = self.tile_managers[scene].groups
            if self.cfg.num_classes:  # check consistency
                if set(groups.keys()) != set(self.cfg.num_classes.keys()):
                    raise ValueError(
                        "Inconsistent groups: "
                        f"{groups.keys()} {self.cfg.num_classes.keys()}"
                    )
                for k in groups:
                    if len(groups[k]) != self.cfg.num_classes[k]:
                        raise ValueError(
                            f"{k}: {len(groups[k])} vs {self.cfg.num_classes[k]}"
                        )
            ppm = self.tile_managers[scene].ppm
            if ppm != self.cfg.pixel_per_meter:
                raise ValueError(
                    "The tile manager and the config/model have different ground "
                    f"resolutions: {ppm} vs {self.cfg.pixel_per_meter}"
                )

            logger.info("Loading dump json file %s.", self.dump_filename)
            with (dump_dir / self.dump_filename).open("r") as fp:
                self.dumps[scene] = pack_dump_dict(json.load(fp))
            for seq, per_seq in self.dumps[scene].items():
                for cam_id, cam_dict in per_seq["cameras"].items():
                    if cam_dict["model"] != "PINHOLE":
                        raise ValueError(
                            "Unsupported camera model: "
                            f"{cam_dict['model']} for {scene},{seq},{cam_id}"
                        )

            self.image_dirs[scene] = (
                # (self.local_dir or self.root) / scene / self.images_dirname
                self.noise_root / scene / self.images_dirname
            )
            assert self.image_dirs[scene].exists(), self.image_dirs[scene]

            for seq, data in self.dumps[scene].items():
                for name in data["views"]:
                    names.append((scene, seq, name))

        self.parse_splits(self.cfg.split, names)
        if self.cfg.filter_for is not None:
            self.filter_elements()
        self.pack_data()

    def pack_data(self):
        # We pack the data into compact tensors
        # that can be shared across processes without copy.
        # We group views by panorama to enable panorama-based batching.
        exclude = {
            "compass_angle",
            "compass_accuracy",
            "gps_accuracy",
            "chunk_key",
            "panorama_offset",
        }
        cameras = {
            scene: {seq: per_seq["cameras"] for seq, per_seq in per_scene.items()}
            for scene, per_scene in self.dumps.items()
        }
        points = {
            scene: {
                seq: {
                    i: torch.from_numpy(p) for i, p in per_seq.get("points", {}).items()
                }
                for seq, per_seq in per_scene.items()
            }
            for scene, per_scene in self.dumps.items()
        }
        self.data = {}
        self.panorama_groups = {}  # Store panorama grouping info

        for stage, names in self.splits.items():
            # Group views by panorama ID
            panorama_views = self._group_views_by_panorama(names)

            # Validate that we have complete panoramas (n views each)
            incomplete_panoramas = [
                p_id for p_id, views in panorama_views.items() if len(views) != self.num_views
            ]
            if incomplete_panoramas:
                logger.warning(
                    f"Found {len(incomplete_panoramas)} incomplete panoramas "
                    f"in {stage} set. These will be excluded. "
                    f"First few: {incomplete_panoramas[:5]}"
                )
                # Keep only complete panoramas
                panorama_views = {
                    p_id: views
                    for p_id, views in panorama_views.items()
                    if len(views) == self.num_views
                }

            # Store panorama groups for this stage
            self.panorama_groups[stage] = panorama_views

            # Create flattened names list maintaining panorama view order
            ordered_names = []
            for p_id in sorted(panorama_views.keys()):
                views = panorama_views[p_id]
                # Sort views by suffix for consistent order:
                # suffix_order = {"front": 0, "left": 1, "back": 2, "right": 3}
                # suffix_order = {"view1": 0, "view2": 1, "view3": 2}
                suffix_order = {"view1": 0, "view2": 1, "view3": 2}
                # suffix_order = {"v1": 0, "v2": 1, "v3": 2, "v4": 3, "v5": 4, "v6": 5, "v7": 6, "v8": 7, "v9": 8}

                views_sorted = sorted(
                    views, key=lambda x: suffix_order.get(x[2].split("_")[-1], 999)
                )
                ordered_names.extend(views_sorted)

            # Pack data as before but with grouped panorama ordering
            if ordered_names:
                view = self.dumps[ordered_names[0][0]][ordered_names[0][1]]["views"][
                    ordered_names[0][2]
                ]
                data = {k: [] for k in view.keys() - exclude}
                for scene, seq, name in ordered_names:
                    for k in data:
                        data[k].append(
                            self.dumps[scene][seq]["views"][name].get(k, None)
                        )
                for k in data:
                    v = np.array(data[k])
                    if np.issubdtype(v.dtype, np.integer) or np.issubdtype(
                        v.dtype, np.floating
                    ):
                        v = torch.from_numpy(v)
                    data[k] = v
                data["cameras"] = cameras
                data["points"] = points
                self.data[stage] = data
                self.splits[stage] = np.array(ordered_names)
            else:
                # Empty stage
                self.data[stage] = {"cameras": cameras, "points": points}
                self.splits[stage] = np.array([])

    def _group_views_by_panorama(self, names):
        """Group view names by panorama ID.

        Args:
            names: List of (scene, seq, name) tuples

        Returns:
            Dict mapping panorama_id to list of (scene, seq, name) tuples
        """
        panorama_views = defaultdict(list)
        for scene, seq, name in names:
            # Extract panorama ID from view name (everything before the last underscore)
            if "_" in name:
                panorama_id = name.rsplit("_", 1)[0]
                panorama_views[panorama_id].append((scene, seq, name))
            else:
                # Handle non-panoramic views if any exist
                panorama_views[name].append((scene, seq, name))
        return dict(panorama_views)

    def filter_elements(self):
        for stage, names in self.splits.items():
            names_select = []
            for scene, seq, name in names:
                view = self.dumps[scene][seq]["views"][name]
                if self.cfg.filter_for == "ground_plane":
                    if not (1.0 <= view["height"] <= 3.0):
                        continue
                    planes = self.dumps[scene][seq].get("plane")
                    if planes is not None:
                        inliers = planes[str(view["chunk_id"])][-1]
                        if inliers < 10:
                            continue
                    if self.cfg.filter_by_ground_angle is not None:
                        plane = np.array(view["plane_params"])
                        normal = plane[:3] / np.linalg.norm(plane[:3])
                        angle = np.rad2deg(np.arccos(np.abs(normal[-1])))
                        if angle > self.cfg.filter_by_ground_angle:
                            continue
                elif self.cfg.filter_for == "pointcloud":
                    if len(view["observations"]) < self.cfg.min_num_points:
                        continue
                elif self.cfg.filter_for is not None:
                    raise ValueError(f"Unknown filtering: {self.cfg.filter_for}")
                names_select.append((scene, seq, name))
            logger.info(
                "%s: Keep %d/%d images after filtering for %s.",
                stage,
                len(names_select),
                len(names),
                self.cfg.filter_for,
            )
            self.splits[stage] = names_select

    def parse_splits(self, split_arg, names):
        if split_arg is None:
            self.splits = {
                "train": names,
                "val": names,
            }
        elif isinstance(split_arg, int):
            names = np.random.RandomState(self.cfg.seed).permutation(names).tolist()
            self.splits = {
                "train": names[split_arg:],
                "val": names[:split_arg],
            }
        elif isinstance(split_arg, DictConfig):
            scenes_val = set(split_arg.val)
            scenes_train = set(split_arg.train)
            assert len(scenes_val - set(self.cfg.scenes)) == 0
            assert len(scenes_train - set(self.cfg.scenes)) == 0
            self.splits = {
                "train": [n for n in names if n[0] in scenes_train],
                "val": [n for n in names if n[0] in scenes_val],
            }
        elif isinstance(split_arg, str):
            if (self.root / split_arg).exists():
                # Common split file.
                with (self.root / split_arg).open("r") as fp:
                    splits = json.load(fp)
            else:
                # Per-scene split file.
                splits = defaultdict(dict)
                for scene in self.cfg.scenes:
                    with (self.root / split_arg.format(scene=scene)).open("r") as fp:
                        scene_splits = json.load(fp)
                    for split_name in scene_splits:
                        splits[split_name][scene] = scene_splits[split_name]
            splits = {
                split_name: {scene: set(ids) for scene, ids in split.items()}
                for split_name, split in splits.items()
            }
            self.splits = {}
            for split_name, split in splits.items():
                self.splits[split_name] = [
                    (scene, *arg, name)
                    for scene, *arg, name in names
                    if scene in split and int(name.rsplit("_", 1)[0]) in split[scene]
                ]
        else:
            raise ValueError(split_arg)

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.splits[stage],
            self.data[stage],
            self.image_dirs,
            self.tile_managers,
            image_ext=".jpg",
        )

    def dataloader(
        self,
        stage: str,
        shuffle: bool = False,
        num_workers: int = None,
        sampler: Optional[torchdata.Sampler] = None,
    ):
        dataset = self.dataset(stage)
        cfg = self.cfg["loading"][stage]
        batch_size = cfg["batch_size"]

        # Validate batch size is a multiple of views per panorama (n)
        # views_per_panorama = self.num_views // 2
        # views_per_pair = 2
        
        if batch_size % self.num_views != 0:
            raise ValueError(
                f"Batch size ({batch_size}) must be a multiple of views per panorama ({self.num_views})."
            )
        
        num_workers = cfg["num_workers"] if num_workers is None else num_workers

        is_distributed = dist.is_available() and dist.is_initialized()
        if stage == 'train' and is_distributed:
            sampler = DistributedGroupSampler(
                dataset,
                samples_per_group = self.num_views,
                shuffle = True,
            )
            shuffle = False # shuffle of Dataloader should be False when sampler is not None
        else:
            shuffle = shuffle

        loader = torchdata.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)

    def sequence_dataset(self, stage: str, **kwargs):
        keys = self.splits[stage]
        seq2indices = defaultdict(list)
        for index, (_, seq, _) in enumerate(keys):
            seq2indices[seq].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(self.data[stage], indices, **kwargs)
            for i, sub_indices in enumerate(chunks):
                chunk2indices[seq, i] = sub_indices
        # store the index of each chunk in its sequence
        chunk_indices = torch.full((len(keys),), -1)
        for (_, chunk_index), idx in chunk2indices.items():
            chunk_indices[idx] = chunk_index
        self.data[stage]["chunk_index"] = chunk_indices
        dataset = self.dataset(stage)
        return dataset, chunk2indices

    def sequence_dataloader(self, stage: str, shuffle: bool = False, **kwargs):
        dataset, chunk2idx = self.sequence_dataset(stage, **kwargs)
        chunk_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(chunk_keys))
            chunk_keys = [chunk_keys[i] for i in perm]
        key_indices = [i for key in chunk_keys for i in chunk2idx[key]]
        num_workers = self.cfg["loading"][stage]["num_workers"]
        loader = torchdata.DataLoader(
            dataset,
            batch_size=None,
            sampler=key_indices,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return loader, chunk_keys, chunk2idx

class MapillaryPanoDataModuleOrigin(pl.LightningDataModule):
    dump_filename = "dump.json"
    # images_archive = "images.tar.gz"
    images_dirname = "images/"

    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "mapillary",
        # paths and fetch
        "data_dir": DATASETS_PATH / "MGL_final",
        "local_dir": None,
        "tiles_filename": "tiles.pkl",
        "scenes": "???",
        "split": None,
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 3, "num_workers": 3},
        },
        "filter_for": None,
        "filter_by_ground_angle": None,
        "min_num_points": "???",
    }

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.num_views = 3
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.local_dir = self.cfg.local_dir or os.environ.get("TMPDIR")
        if self.local_dir is not None:
            self.local_dir = Path(self.local_dir, "MGL_final")
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")

    def prepare_data(self):
        for scene in self.cfg.scenes:
            dump_dir = self.root / scene
            assert (dump_dir / self.dump_filename).exists(), dump_dir
            assert (dump_dir / self.cfg.tiles_filename).exists(), dump_dir
            if self.local_dir is None:
                assert (dump_dir / self.images_dirname).exists(), dump_dir
                continue
            # Cache the folder of images locally to speed up reading
            local_dir = self.local_dir / scene
            if local_dir.exists():
                shutil.rmtree(local_dir)
            local_dir.mkdir(exist_ok=True, parents=True)
            # images_archive = dump_dir / self.images_archive
            # logger.info("Extracting the image archive %s.", images_archive)
            # with tarfile.open(images_archive) as fp:
                # fp.extractall(local_dir)

    def setup(self, stage: Optional[str] = None):
        self.dumps = {}
        self.tile_managers = {}
        self.image_dirs = {}
        names = []

        for scene in self.cfg.scenes:
            logger.info("Loading scene %s.", scene)
            dump_dir = self.root / scene
            logger.info("Loading map tiles %s.", self.cfg.tiles_filename)
            self.tile_managers[scene] = TileManager.load(
                dump_dir / self.cfg.tiles_filename
            )
            groups = self.tile_managers[scene].groups
            if self.cfg.num_classes:  # check consistency
                if set(groups.keys()) != set(self.cfg.num_classes.keys()):
                    raise ValueError(
                        "Inconsistent groups: "
                        f"{groups.keys()} {self.cfg.num_classes.keys()}"
                    )
                for k in groups:
                    if len(groups[k]) != self.cfg.num_classes[k]:
                        raise ValueError(
                            f"{k}: {len(groups[k])} vs {self.cfg.num_classes[k]}"
                        )
            ppm = self.tile_managers[scene].ppm
            if ppm != self.cfg.pixel_per_meter:
                raise ValueError(
                    "The tile manager and the config/model have different ground "
                    f"resolutions: {ppm} vs {self.cfg.pixel_per_meter}"
                )

            logger.info("Loading dump json file %s.", self.dump_filename)
            with (dump_dir / self.dump_filename).open("r") as fp:
                self.dumps[scene] = pack_dump_dict(json.load(fp))
            for seq, per_seq in self.dumps[scene].items():
                for cam_id, cam_dict in per_seq["cameras"].items():
                    if cam_dict["model"] != "PINHOLE":
                        raise ValueError(
                            "Unsupported camera model: "
                            f"{cam_dict['model']} for {scene},{seq},{cam_id}"
                        )

            self.image_dirs[scene] = (
                # (self.local_dir or self.root) / scene / self.images_dirname
                self.root / scene / self.images_dirname
            )
            assert self.image_dirs[scene].exists(), self.image_dirs[scene]

            for seq, data in self.dumps[scene].items():
                for name in data["views"]:
                    names.append((scene, seq, name))

        self.parse_splits(self.cfg.split, names)
        if self.cfg.filter_for is not None:
            self.filter_elements()
        self.pack_data()

    def pack_data(self):
        # We pack the data into compact tensors
        # that can be shared across processes without copy.
        # We group views by panorama to enable panorama-based batching.
        exclude = {
            "compass_angle",
            "compass_accuracy",
            "gps_accuracy",
            "chunk_key",
            "panorama_offset",
        }
        cameras = {
            scene: {seq: per_seq["cameras"] for seq, per_seq in per_scene.items()}
            for scene, per_scene in self.dumps.items()
        }
        points = {
            scene: {
                seq: {
                    i: torch.from_numpy(p) for i, p in per_seq.get("points", {}).items()
                }
                for seq, per_seq in per_scene.items()
            }
            for scene, per_scene in self.dumps.items()
        }
        self.data = {}
        self.panorama_groups = {}  # Store panorama grouping info

        for stage, names in self.splits.items():
            # Group views by panorama ID
            panorama_views = self._group_views_by_panorama(names)

            # Validate that we have complete panoramas (n views each)
            incomplete_panoramas = [
                p_id for p_id, views in panorama_views.items() if len(views) != self.num_views
            ]
            if incomplete_panoramas:
                logger.warning(
                    f"Found {len(incomplete_panoramas)} incomplete panoramas "
                    f"in {stage} set. These will be excluded. "
                    f"First few: {incomplete_panoramas[:5]}"
                )
                # Keep only complete panoramas
                panorama_views = {
                    p_id: views
                    for p_id, views in panorama_views.items()
                    if len(views) == self.num_views
                }

            # Store panorama groups for this stage
            self.panorama_groups[stage] = panorama_views

            # Create flattened names list maintaining panorama view order
            ordered_names = []
            for p_id in sorted(panorama_views.keys()):
                views = panorama_views[p_id]
                # Sort views by suffix for consistent order:
                # suffix_order = {"front": 0, "left": 1, "back": 2, "right": 3}
                suffix_order = {"view1": 0, "view2": 1, "view3": 2}
                views_sorted = sorted(
                    views, key=lambda x: suffix_order.get(x[2].split("_")[-1], 999)
                )
                ordered_names.extend(views_sorted)

            # Pack data as before but with grouped panorama ordering
            if ordered_names:
                view = self.dumps[ordered_names[0][0]][ordered_names[0][1]]["views"][
                    ordered_names[0][2]
                ]
                data = {k: [] for k in view.keys() - exclude}
                for scene, seq, name in ordered_names:
                    for k in data:
                        data[k].append(
                            self.dumps[scene][seq]["views"][name].get(k, None)
                        )
                for k in data:
                    v = np.array(data[k])
                    if np.issubdtype(v.dtype, np.integer) or np.issubdtype(
                        v.dtype, np.floating
                    ):
                        v = torch.from_numpy(v)
                    data[k] = v
                data["cameras"] = cameras
                data["points"] = points
                self.data[stage] = data
                self.splits[stage] = np.array(ordered_names)
            else:
                # Empty stage
                self.data[stage] = {"cameras": cameras, "points": points}
                self.splits[stage] = np.array([])

    def _group_views_by_panorama(self, names):
        """Group view names by panorama ID.

        Args:
            names: List of (scene, seq, name) tuples

        Returns:
            Dict mapping panorama_id to list of (scene, seq, name) tuples
        """
        panorama_views = defaultdict(list)
        for scene, seq, name in names:
            # Extract panorama ID from view name (everything before the last underscore)
            if "_" in name:
                panorama_id = name.rsplit("_", 1)[0]
                panorama_views[panorama_id].append((scene, seq, name))
            else:
                # Handle non-panoramic views if any exist
                panorama_views[name].append((scene, seq, name))
        return dict(panorama_views)

    def filter_elements(self):
        for stage, names in self.splits.items():
            names_select = []
            for scene, seq, name in names:
                view = self.dumps[scene][seq]["views"][name]
                if self.cfg.filter_for == "ground_plane":
                    if not (1.0 <= view["height"] <= 3.0):
                        continue
                    planes = self.dumps[scene][seq].get("plane")
                    if planes is not None:
                        inliers = planes[str(view["chunk_id"])][-1]
                        if inliers < 10:
                            continue
                    if self.cfg.filter_by_ground_angle is not None:
                        plane = np.array(view["plane_params"])
                        normal = plane[:3] / np.linalg.norm(plane[:3])
                        angle = np.rad2deg(np.arccos(np.abs(normal[-1])))
                        if angle > self.cfg.filter_by_ground_angle:
                            continue
                elif self.cfg.filter_for == "pointcloud":
                    if len(view["observations"]) < self.cfg.min_num_points:
                        continue
                elif self.cfg.filter_for is not None:
                    raise ValueError(f"Unknown filtering: {self.cfg.filter_for}")
                names_select.append((scene, seq, name))
            logger.info(
                "%s: Keep %d/%d images after filtering for %s.",
                stage,
                len(names_select),
                len(names),
                self.cfg.filter_for,
            )
            self.splits[stage] = names_select

    def parse_splits(self, split_arg, names):
        if split_arg is None:
            self.splits = {
                "train": names,
                "val": names,
            }
        elif isinstance(split_arg, int):
            names = np.random.RandomState(self.cfg.seed).permutation(names).tolist()
            self.splits = {
                "train": names[split_arg:],
                "val": names[:split_arg],
            }
        elif isinstance(split_arg, DictConfig):
            scenes_val = set(split_arg.val)
            scenes_train = set(split_arg.train)
            assert len(scenes_val - set(self.cfg.scenes)) == 0
            assert len(scenes_train - set(self.cfg.scenes)) == 0
            self.splits = {
                "train": [n for n in names if n[0] in scenes_train],
                "val": [n for n in names if n[0] in scenes_val],
            }
        elif isinstance(split_arg, str):
            if (self.root / split_arg).exists():
                # Common split file.
                with (self.root / split_arg).open("r") as fp:
                    splits = json.load(fp)
            else:
                # Per-scene split file.
                splits = defaultdict(dict)
                for scene in self.cfg.scenes:
                    with (self.root / split_arg.format(scene=scene)).open("r") as fp:
                        scene_splits = json.load(fp)
                    for split_name in scene_splits:
                        splits[split_name][scene] = scene_splits[split_name]
            splits = {
                split_name: {scene: set(ids) for scene, ids in split.items()}
                for split_name, split in splits.items()
            }
            self.splits = {}
            for split_name, split in splits.items():
                self.splits[split_name] = [
                    (scene, *arg, name)
                    for scene, *arg, name in names
                    if scene in split and int(name.rsplit("_", 1)[0]) in split[scene]
                ]
        else:
            raise ValueError(split_arg)

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.splits[stage],
            self.data[stage],
            self.image_dirs,
            self.tile_managers,
            image_ext=".jpg",
        )

    def dataloader(
        self,
        stage: str,
        shuffle: bool = False,
        num_workers: int = None,
        sampler: Optional[torchdata.Sampler] = None,
    ):
        dataset = self.dataset(stage)
        cfg = self.cfg["loading"][stage]
        batch_size = cfg["batch_size"]

        # Validate batch size is a multiple of views per panorama (n)
        views_per_panorama = self.num_views
        if batch_size % views_per_panorama != 0:
            raise ValueError(
                f"Batch size ({batch_size}) must be a multiple of views per "
                f"panorama ({views_per_panorama}). Each panorama contains "
                f"{views_per_panorama} views: front, left, back right."
            )

        num_workers = cfg["num_workers"] if num_workers is None else num_workers
        loader = torchdata.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            # shuffle=shuffle or (stage == "train"),
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)

    def sequence_dataset(self, stage: str, **kwargs):
        keys = self.splits[stage]
        seq2indices = defaultdict(list)
        for index, (_, seq, _) in enumerate(keys):
            seq2indices[seq].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(self.data[stage], indices, **kwargs)
            for i, sub_indices in enumerate(chunks):
                chunk2indices[seq, i] = sub_indices
        # store the index of each chunk in its sequence
        chunk_indices = torch.full((len(keys),), -1)
        for (_, chunk_index), idx in chunk2indices.items():
            chunk_indices[idx] = chunk_index
        self.data[stage]["chunk_index"] = chunk_indices
        dataset = self.dataset(stage)
        return dataset, chunk2indices

    def sequence_dataloader(self, stage: str, shuffle: bool = False, **kwargs):
        dataset, chunk2idx = self.sequence_dataset(stage, **kwargs)
        chunk_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(chunk_keys))
            chunk_keys = [chunk_keys[i] for i in perm]
        key_indices = [i for key in chunk_keys for i in chunk2idx[key]]
        num_workers = self.cfg["loading"][stage]["num_workers"]
        loader = torchdata.DataLoader(
            dataset,
            batch_size=None,
            sampler=key_indices,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return loader, chunk_keys, chunk2idx

class MapillaryDataModule(pl.LightningDataModule):
    dump_filename = "dump.json"
    # images_archive = "images.tar.gz"
    images_dirname = "images/"

    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "mapillary",
        # paths and fetch
        "data_dir": DATASETS_PATH / "Sim2Real",
        "local_dir": None,
        "tiles_filename": "tiles.pkl",
        "scenes": "???",
        "split": None,
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 1, "num_workers": 0},
        },
        "filter_for": None,
        "filter_by_ground_angle": None,
        "min_num_points": "???",
    }

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.local_dir = self.cfg.local_dir or os.environ.get("TMPDIR")
        if self.local_dir is not None:
            self.local_dir = Path(self.local_dir, "Sim2Real")
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")

    def prepare_data(self):
        for scene in self.cfg.scenes:
            dump_dir = self.root / scene
            assert (dump_dir / self.dump_filename).exists(), dump_dir
            assert (dump_dir / self.cfg.tiles_filename).exists(), dump_dir
            if self.local_dir is None:
                assert (dump_dir / self.images_dirname).exists(), dump_dir
                continue
            # Cache the folder of images locally to speed up reading
            local_dir = self.local_dir / scene
            if local_dir.exists():
                shutil.rmtree(local_dir)
            local_dir.mkdir(exist_ok=True, parents=True)
            # images_archive = dump_dir / self.images_archive
            # logger.info("Extracting the image archive %s.", images_archive)
            # with tarfile.open(images_archive) as fp:
                # fp.extractall(local_dir)

    def setup(self, stage: Optional[str] = None):
        self.dumps = {}
        self.tile_managers = {}
        self.image_dirs = {}
        names = []

        for scene in self.cfg.scenes:
            logger.info("Loading scene %s.", scene)
            dump_dir = self.root / scene
            logger.info("Loading map tiles %s.", self.cfg.tiles_filename)
            self.tile_managers[scene] = TileManager.load(
                dump_dir / self.cfg.tiles_filename
            )
            groups = self.tile_managers[scene].groups
            if self.cfg.num_classes:  # check consistency
                if set(groups.keys()) != set(self.cfg.num_classes.keys()):
                    raise ValueError(
                        "Inconsistent groups: "
                        f"{groups.keys()} {self.cfg.num_classes.keys()}"
                    )
                for k in groups:
                    if len(groups[k]) != self.cfg.num_classes[k]:
                        raise ValueError(
                            f"{k}: {len(groups[k])} vs {self.cfg.num_classes[k]}"
                        )
            ppm = self.tile_managers[scene].ppm
            if ppm != self.cfg.pixel_per_meter:
                raise ValueError(
                    "The tile manager and the config/model have different ground "
                    f"resolutions: {ppm} vs {self.cfg.pixel_per_meter}"
                )

            logger.info("Loading dump json file %s.", self.dump_filename)
            with (dump_dir / self.dump_filename).open("r") as fp:
                self.dumps[scene] = pack_dump_dict(json.load(fp))
            for seq, per_seq in self.dumps[scene].items():
                for cam_id, cam_dict in per_seq["cameras"].items():
                    if cam_dict["model"] != "PINHOLE":
                        raise ValueError(
                            "Unsupported camera model: "
                            f"{cam_dict['model']} for {scene},{seq},{cam_id}"
                        )

            self.image_dirs[scene] = (
                # (self.local_dir or self.root) / scene / self.images_dirname
                self.root / scene / self.images_dirname
            )
            assert self.image_dirs[scene].exists(), self.image_dirs[scene]

            for seq, data in self.dumps[scene].items():
                for name in data["views"]:
                    names.append((scene, seq, name))

        self.parse_splits(self.cfg.split, names)
        if self.cfg.filter_for is not None:
            self.filter_elements()
        self.pack_data()

    def pack_data(self):
        # We pack the data into compact tensors
        # that can be shared across processes without copy.
        exclude = {
            "compass_angle",
            "compass_accuracy",
            "gps_accuracy",
            "chunk_key",
            "panorama_offset",
        }
        cameras = {
            scene: {seq: per_seq["cameras"] for seq, per_seq in per_scene.items()}
            for scene, per_scene in self.dumps.items()
        }
        points = {
            scene: {
                seq: {
                    i: torch.from_numpy(p) for i, p in per_seq.get("points", {}).items()
                }
                for seq, per_seq in per_scene.items()
            }
            for scene, per_scene in self.dumps.items()
        }
        self.data = {}
        for stage, names in self.splits.items():
            view = self.dumps[names[0][0]][names[0][1]]["views"][names[0][2]]
            data = {k: [] for k in view.keys() - exclude}
            for scene, seq, name in names:
                for k in data:
                    data[k].append(self.dumps[scene][seq]["views"][name].get(k, None))
            for k in data:
                v = np.array(data[k])
                if np.issubdtype(v.dtype, np.integer) or np.issubdtype(
                    v.dtype, np.floating
                ):
                    v = torch.from_numpy(v)
                data[k] = v
            data["cameras"] = cameras
            data["points"] = points
            self.data[stage] = data
            self.splits[stage] = np.array(names)

    def filter_elements(self):
        for stage, names in self.splits.items():
            names_select = []
            for scene, seq, name in names:
                view = self.dumps[scene][seq]["views"][name]
                if self.cfg.filter_for == "ground_plane":
                    if not (1.0 <= view["height"] <= 3.0):
                        continue
                    planes = self.dumps[scene][seq].get("plane")
                    if planes is not None:
                        inliers = planes[str(view["chunk_id"])][-1]
                        if inliers < 10:
                            continue
                    if self.cfg.filter_by_ground_angle is not None:
                        plane = np.array(view["plane_params"])
                        normal = plane[:3] / np.linalg.norm(plane[:3])
                        angle = np.rad2deg(np.arccos(np.abs(normal[-1])))
                        if angle > self.cfg.filter_by_ground_angle:
                            continue
                elif self.cfg.filter_for == "pointcloud":
                    if len(view["observations"]) < self.cfg.min_num_points:
                        continue
                elif self.cfg.filter_for is not None:
                    raise ValueError(f"Unknown filtering: {self.cfg.filter_for}")
                names_select.append((scene, seq, name))
            logger.info(
                "%s: Keep %d/%d images after filtering for %s.",
                stage,
                len(names_select),
                len(names),
                self.cfg.filter_for,
            )
            self.splits[stage] = names_select

    def parse_splits(self, split_arg, names):
        if split_arg is None:
            self.splits = {
                "train": names,
                "val": names,
            }
        elif isinstance(split_arg, int):
            names = np.random.RandomState(self.cfg.seed).permutation(names).tolist()
            self.splits = {
                "train": names[split_arg:],
                "val": names[:split_arg],
            }
        elif isinstance(split_arg, DictConfig):
            scenes_val = set(split_arg.val)
            scenes_train = set(split_arg.train)
            assert len(scenes_val - set(self.cfg.scenes)) == 0
            assert len(scenes_train - set(self.cfg.scenes)) == 0
            self.splits = {
                "train": [n for n in names if n[0] in scenes_train],
                "val": [n for n in names if n[0] in scenes_val],
            }
        elif isinstance(split_arg, str):
            if (self.root / split_arg).exists():
                # Common split file.
                with (self.root / split_arg).open("r") as fp:
                    splits = json.load(fp)
            else:
                # Per-scene split file.
                splits = defaultdict(dict)
                for scene in self.cfg.scenes:
                    with (self.root / split_arg.format(scene=scene)).open("r") as fp:
                        scene_splits = json.load(fp)
                    for split_name in scene_splits:
                        splits[split_name][scene] = scene_splits[split_name]
            splits = {
                split_name: {scene: set(ids) for scene, ids in split.items()}
                for split_name, split in splits.items()
            }
            self.splits = {}
            for split_name, split in splits.items():
                self.splits[split_name] = [
                    (scene, *arg, name)
                    for scene, *arg, name in names
                    if scene in split and int(name.rsplit("_", 1)[0]) in split[scene]
                ]
        else:
            raise ValueError(split_arg)

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.splits[stage],
            self.data[stage],
            self.image_dirs,
            self.tile_managers,
            image_ext=".jpg",
        )

    def dataloader(
        self,
        stage: str,
        shuffle: bool = False,
        num_workers: int = None,
        sampler: Optional[torchdata.Sampler] = None,
    ):
        dataset = self.dataset(stage)
        cfg = self.cfg["loading"][stage]
        num_workers = cfg["num_workers"] if num_workers is None else num_workers
        loader = torchdata.DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=num_workers,
            shuffle=shuffle or (stage == "train"),
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)

    def sequence_dataset(self, stage: str, **kwargs):
        keys = self.splits[stage]
        seq2indices = defaultdict(list)
        for index, (_, seq, _) in enumerate(keys):
            seq2indices[seq].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(self.data[stage], indices, **kwargs)
            for i, sub_indices in enumerate(chunks):
                chunk2indices[seq, i] = sub_indices
        # store the index of each chunk in its sequence
        chunk_indices = torch.full((len(keys),), -1)
        for (_, chunk_index), idx in chunk2indices.items():
            chunk_indices[idx] = chunk_index
        self.data[stage]["chunk_index"] = chunk_indices
        dataset = self.dataset(stage)
        return dataset, chunk2indices

    def sequence_dataloader(self, stage: str, shuffle: bool = False, **kwargs):
        dataset, chunk2idx = self.sequence_dataset(stage, **kwargs)
        chunk_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(chunk_keys))
            chunk_keys = [chunk_keys[i] for i in perm]
        key_indices = [i for key in chunk_keys for i in chunk2idx[key]]
        num_workers = self.cfg["loading"][stage]["num_workers"]
        loader = torchdata.DataLoader(
            dataset,
            batch_size=None,
            sampler=key_indices,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return loader, chunk_keys, chunk2idx

class MapillaryMixedDataModule(pl.LightningDataModule):
    dump_filename = "dump.json"
    # images_archive = "images.tar.gz"
    images_dirname = "images/"

    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "mapillary",
        # paths and fetch
        "data_dir": DATASETS_PATH,
        "local_dir": None,
        "tiles_filename": "tiles.pkl",
        "scenes": "???",
        "split": None,
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 1, "num_workers": 0},
        },
        "filter_for": None,
        "filter_by_ground_angle": None,
        "min_num_points": "???",
    }

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        #self.local_dir = self.cfg.local_dir or os.environ.get("TMPDIR")
        self.local_dir = self.cfg.local_dir
        if self.local_dir is not None:
            self.local_dir = Path(self.local_dir, "OrienterNet/datasets")
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")

    def prepare_data(self):
        for scene in self.cfg.scenes:
            dump_dir = self.root / scene
            assert (dump_dir / self.dump_filename).exists(), dump_dir
            assert (dump_dir / self.cfg.tiles_filename).exists(), dump_dir
            if self.local_dir is None:
                assert (dump_dir / self.images_dirname).exists(), dump_dir
                continue
            # Cache the folder of images locally to speed up reading
            local_dir = self.local_dir / scene
            if local_dir.exists():
                shutil.rmtree(local_dir)
            local_dir.mkdir(exist_ok=True, parents=True)
            # images_archive = dump_dir / self.images_archive
            # logger.info("Extracting the image archive %s.", images_archive)
            # with tarfile.open(images_archive) as fp:
                # fp.extractall(local_dir)

    def setup(self, stage: Optional[str] = None):
        self.dumps = {}
        self.tile_managers = {}
        self.image_dirs = {}
        names = []

        for scene in self.cfg.scenes:
            logger.info("Loading scene %s.", scene)
            dump_dir = self.root / scene
            logger.info("Loading map tiles %s.", self.cfg.tiles_filename)
            self.tile_managers[scene] = TileManager.load(
                dump_dir / self.cfg.tiles_filename
            )
            groups = self.tile_managers[scene].groups
            if self.cfg.num_classes:  # check consistency
                if set(groups.keys()) != set(self.cfg.num_classes.keys()):
                    raise ValueError(
                        "Inconsistent groups: "
                        f"{groups.keys()} {self.cfg.num_classes.keys()}"
                    )
                for k in groups:
                    if len(groups[k]) != self.cfg.num_classes[k]:
                        raise ValueError(
                            f"{k}: {len(groups[k])} vs {self.cfg.num_classes[k]}"
                        )
            ppm = self.tile_managers[scene].ppm
            if ppm != self.cfg.pixel_per_meter:
                raise ValueError(
                    "The tile manager and the config/model have different ground "
                    f"resolutions: {ppm} vs {self.cfg.pixel_per_meter}"
                )

            logger.info("Loading dump json file %s.", self.dump_filename)
            with (dump_dir / self.dump_filename).open("r") as fp:
                self.dumps[scene] = pack_dump_dict(json.load(fp))
            for seq, per_seq in self.dumps[scene].items():
                for cam_id, cam_dict in per_seq["cameras"].items():
                    if cam_dict["model"] != "PINHOLE":
                        raise ValueError(
                            "Unsupported camera model: "
                            f"{cam_dict['model']} for {scene},{seq},{cam_id}"
                        )

            self.image_dirs[scene] = (
                # (self.local_dir or self.root) / scene / self.images_dirname
                self.root / scene / self.images_dirname
            )
            assert self.image_dirs[scene].exists(), self.image_dirs[scene]

            for seq, data in self.dumps[scene].items():
                for name in data["views"]:
                    names.append((scene, seq, name))

        self.parse_splits(self.cfg.split, names)
        if self.cfg.filter_for is not None:
            self.filter_elements()
        self.pack_data()

    def pack_data(self):
        # We pack the data into compact tensors
        # that can be shared across processes without copy.
        exclude = {
            "compass_angle",
            "compass_accuracy",
            "gps_accuracy",
            "chunk_key",
            "panorama_offset",
        }
        cameras = {
            scene: {seq: per_seq["cameras"] for seq, per_seq in per_scene.items()}
            for scene, per_scene in self.dumps.items()
        }
        points = {
            scene: {
                seq: {
                    i: torch.from_numpy(p) for i, p in per_seq.get("points", {}).items()
                }
                for seq, per_seq in per_scene.items()
            }
            for scene, per_scene in self.dumps.items()
        }
        self.data = {}
        for stage, names in self.splits.items():
            view = self.dumps[names[0][0]][names[0][1]]["views"][names[0][2]]
            data = {k: [] for k in view.keys() - exclude}
            for scene, seq, name in names:
                for k in data:
                    data[k].append(self.dumps[scene][seq]["views"][name].get(k, None))
            for k in data:
                v = np.array(data[k])
                if np.issubdtype(v.dtype, np.integer) or np.issubdtype(
                    v.dtype, np.floating
                ):
                    v = torch.from_numpy(v)
                data[k] = v
            data["cameras"] = cameras
            data["points"] = points
            self.data[stage] = data
            self.splits[stage] = np.array(names)

    def filter_elements(self):
        for stage, names in self.splits.items():
            names_select = []
            for scene, seq, name in names:
                view = self.dumps[scene][seq]["views"][name]
                if self.cfg.filter_for == "ground_plane":
                    if not (1.0 <= view["height"] <= 3.0):
                        continue
                    planes = self.dumps[scene][seq].get("plane")
                    if planes is not None:
                        inliers = planes[str(view["chunk_id"])][-1]
                        if inliers < 10:
                            continue
                    if self.cfg.filter_by_ground_angle is not None:
                        plane = np.array(view["plane_params"])
                        normal = plane[:3] / np.linalg.norm(plane[:3])
                        angle = np.rad2deg(np.arccos(np.abs(normal[-1])))
                        if angle > self.cfg.filter_by_ground_angle:
                            continue
                elif self.cfg.filter_for == "pointcloud":
                    if len(view["observations"]) < self.cfg.min_num_points:
                        continue
                elif self.cfg.filter_for is not None:
                    raise ValueError(f"Unknown filtering: {self.cfg.filter_for}")
                names_select.append((scene, seq, name))
            logger.info(
                "%s: Keep %d/%d images after filtering for %s.",
                stage,
                len(names_select),
                len(names),
                self.cfg.filter_for,
            )
            self.splits[stage] = names_select

    def parse_splits(self, split_arg, names):
        if split_arg is None:
            self.splits = {
                "train": names,
                "val": names,
            }
        elif isinstance(split_arg, int):
            names = np.random.RandomState(self.cfg.seed).permutation(names).tolist()
            self.splits = {
                "train": names[split_arg:],
                "val": names[:split_arg],
            }
        elif isinstance(split_arg, DictConfig):
            scenes_val = set(split_arg.val)
            scenes_train = set(split_arg.train)
            assert len(scenes_val - set(self.cfg.scenes)) == 0
            assert len(scenes_train - set(self.cfg.scenes)) == 0
            self.splits = {
                "train": [n for n in names if n[0] in scenes_train],
                "val": [n for n in names if n[0] in scenes_val],
            }
        elif isinstance(split_arg, str):
            if (self.root / split_arg).exists():
                # Common split file.
                with (self.root / split_arg).open("r") as fp:
                    splits = json.load(fp)
            else:
                # Per-scene split file.
                splits = defaultdict(dict)
                for scene in self.cfg.scenes:
                    with (self.root / split_arg.format(scene=scene)).open("r") as fp:
                        scene_splits = json.load(fp)
                    for split_name in scene_splits:
                        splits[split_name][scene] = scene_splits[split_name]
            splits = {
                split_name: {scene: set(ids) for scene, ids in split.items()}
                for split_name, split in splits.items()
            }
            self.splits = {}
            for split_name, split in splits.items():
                self.splits[split_name] = [
                    (scene, *arg, name)
                    for scene, *arg, name in names
                    if scene in split and int(name.rsplit("_", 1)[0]) in split[scene]
                ]
        else:
            raise ValueError(split_arg)

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.splits[stage],
            self.data[stage],
            self.image_dirs,
            self.tile_managers,
            image_ext=".jpg",
        )

    def dataloader(
        self,
        stage: str,
        shuffle: bool = False,
        num_workers: int = None,
        sampler: Optional[torchdata.Sampler] = None,
    ):
        dataset = self.dataset(stage)
        cfg = self.cfg["loading"][stage]
        num_workers = cfg["num_workers"] if num_workers is None else num_workers
        loader = torchdata.DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=num_workers,
            shuffle=shuffle or (stage == "train"),
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)

    def sequence_dataset(self, stage: str, **kwargs):
        keys = self.splits[stage]
        seq2indices = defaultdict(list)
        for index, (_, seq, _) in enumerate(keys):
            seq2indices[seq].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(self.data[stage], indices, **kwargs)
            for i, sub_indices in enumerate(chunks):
                chunk2indices[seq, i] = sub_indices
        # store the index of each chunk in its sequence
        chunk_indices = torch.full((len(keys),), -1)
        for (_, chunk_index), idx in chunk2indices.items():
            chunk_indices[idx] = chunk_index
        self.data[stage]["chunk_index"] = chunk_indices
        dataset = self.dataset(stage)
        return dataset, chunk2indices

    def sequence_dataloader(self, stage: str, shuffle: bool = False, **kwargs):
        dataset, chunk2idx = self.sequence_dataset(stage, **kwargs)
        chunk_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(chunk_keys))
            chunk_keys = [chunk_keys[i] for i in perm]
        key_indices = [i for key in chunk_keys for i in chunk2idx[key]]
        num_workers = self.cfg["loading"][stage]["num_workers"]
        loader = torchdata.DataLoader(
            dataset,
            batch_size=None,
            sampler=key_indices,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return loader, chunk_keys, chunk2idx

class Mapillary360DataModule(pl.LightningDataModule):
    dump_filename = "dump.json"
    # images_archive = "images.tar.gz"
    images_dirname = "images/"

    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "mapillary",
        # paths and fetch
        "data_dir": DATASETS_PATH / "MGL_360",
        "local_dir": None,
        "tiles_filename": "tiles.pkl",
        "scenes": "???",
        "split": None,
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 1, "num_workers": 0},
        },
        "filter_for": None,
        "filter_by_ground_angle": None,
        "min_num_points": "???",
    }

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.local_dir = self.cfg.local_dir or os.environ.get("TMPDIR")
        if self.local_dir is not None:
            self.local_dir = Path(self.local_dir, "MGL_360")
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")

    def prepare_data(self):
        for scene in self.cfg.scenes:
            dump_dir = self.root / scene
            assert (dump_dir / self.dump_filename).exists(), dump_dir
            assert (dump_dir / self.cfg.tiles_filename).exists(), dump_dir
            if self.local_dir is None:
                assert (dump_dir / self.images_dirname).exists(), dump_dir
                continue
            # Cache the folder of images locally to speed up reading
            local_dir = self.local_dir / scene
            if local_dir.exists():
                shutil.rmtree(local_dir)
            local_dir.mkdir(exist_ok=True, parents=True)
            # images_archive = dump_dir / self.images_archive
            # logger.info("Extracting the image archive %s.", images_archive)
            # with tarfile.open(images_archive) as fp:
                # fp.extractall(local_dir)

    def setup(self, stage: Optional[str] = None):
        self.dumps = {}
        self.tile_managers = {}
        self.image_dirs = {}
        names = []

        for scene in self.cfg.scenes:
            logger.info("Loading scene %s.", scene)
            dump_dir = self.root / scene
            logger.info("Loading map tiles %s.", self.cfg.tiles_filename)
            self.tile_managers[scene] = TileManager.load(
                dump_dir / self.cfg.tiles_filename
            )
            groups = self.tile_managers[scene].groups
            if self.cfg.num_classes:  # check consistency
                if set(groups.keys()) != set(self.cfg.num_classes.keys()):
                    raise ValueError(
                        "Inconsistent groups: "
                        f"{groups.keys()} {self.cfg.num_classes.keys()}"
                    )
                for k in groups:
                    if len(groups[k]) != self.cfg.num_classes[k]:
                        raise ValueError(
                            f"{k}: {len(groups[k])} vs {self.cfg.num_classes[k]}"
                        )
            ppm = self.tile_managers[scene].ppm
            if ppm != self.cfg.pixel_per_meter:
                raise ValueError(
                    "The tile manager and the config/model have different ground "
                    f"resolutions: {ppm} vs {self.cfg.pixel_per_meter}"
                )

            logger.info("Loading dump json file %s.", self.dump_filename)
            with (dump_dir / self.dump_filename).open("r") as fp:
                self.dumps[scene] = pack_dump_dict(json.load(fp))
            for seq, per_seq in self.dumps[scene].items():
                for cam_id, cam_dict in per_seq["cameras"].items():
                    if cam_dict["model"] != "PINHOLE":
                        raise ValueError(
                            "Unsupported camera model: "
                            f"{cam_dict['model']} for {scene},{seq},{cam_id}"
                        )

            self.image_dirs[scene] = (
                # (self.local_dir or self.root) / scene / self.images_dirname
                self.root / scene / self.images_dirname
            )
            assert self.image_dirs[scene].exists(), self.image_dirs[scene]

            for seq, data in self.dumps[scene].items():
                for name in data["views"]:
                    names.append((scene, seq, name))

        self.parse_splits(self.cfg.split, names)
        if self.cfg.filter_for is not None:
            self.filter_elements()
        self.pack_data()

    def pack_data(self):
        # We pack the data into compact tensors
        # that can be shared across processes without copy.
        exclude = {
            "compass_angle",
            "compass_accuracy",
            "gps_accuracy",
            "chunk_key",
            "panorama_offset",
        }
        cameras = {
            scene: {seq: per_seq["cameras"] for seq, per_seq in per_scene.items()}
            for scene, per_scene in self.dumps.items()
        }
        points = {
            scene: {
                seq: {
                    i: torch.from_numpy(p) for i, p in per_seq.get("points", {}).items()
                }
                for seq, per_seq in per_scene.items()
            }
            for scene, per_scene in self.dumps.items()
        }
        self.data = {}
        for stage, names in self.splits.items():
            view = self.dumps[names[0][0]][names[0][1]]["views"][names[0][2]]
            data = {k: [] for k in view.keys() - exclude}
            for scene, seq, name in names:
                for k in data:
                    data[k].append(self.dumps[scene][seq]["views"][name].get(k, None))
            for k in data:
                v = np.array(data[k])
                if np.issubdtype(v.dtype, np.integer) or np.issubdtype(
                    v.dtype, np.floating
                ):
                    v = torch.from_numpy(v)
                data[k] = v
            data["cameras"] = cameras
            data["points"] = points
            self.data[stage] = data
            self.splits[stage] = np.array(names)

    def filter_elements(self):
        for stage, names in self.splits.items():
            names_select = []
            for scene, seq, name in names:
                view = self.dumps[scene][seq]["views"][name]
                if self.cfg.filter_for == "ground_plane":
                    if not (1.0 <= view["height"] <= 3.0):
                        continue
                    planes = self.dumps[scene][seq].get("plane")
                    if planes is not None:
                        inliers = planes[str(view["chunk_id"])][-1]
                        if inliers < 10:
                            continue
                    if self.cfg.filter_by_ground_angle is not None:
                        plane = np.array(view["plane_params"])
                        normal = plane[:3] / np.linalg.norm(plane[:3])
                        angle = np.rad2deg(np.arccos(np.abs(normal[-1])))
                        if angle > self.cfg.filter_by_ground_angle:
                            continue
                elif self.cfg.filter_for == "pointcloud":
                    if len(view["observations"]) < self.cfg.min_num_points:
                        continue
                elif self.cfg.filter_for is not None:
                    raise ValueError(f"Unknown filtering: {self.cfg.filter_for}")
                names_select.append((scene, seq, name))
            logger.info(
                "%s: Keep %d/%d images after filtering for %s.",
                stage,
                len(names_select),
                len(names),
                self.cfg.filter_for,
            )
            self.splits[stage] = names_select

    def parse_splits(self, split_arg, names):
        if split_arg is None:
            self.splits = {
                "train": names,
                "val": names,
            }
        elif isinstance(split_arg, int):
            names = np.random.RandomState(self.cfg.seed).permutation(names).tolist()
            self.splits = {
                "train": names[split_arg:],
                "val": names[:split_arg],
            }
        elif isinstance(split_arg, DictConfig):
            scenes_val = set(split_arg.val)
            scenes_train = set(split_arg.train)
            assert len(scenes_val - set(self.cfg.scenes)) == 0
            assert len(scenes_train - set(self.cfg.scenes)) == 0
            self.splits = {
                "train": [n for n in names if n[0] in scenes_train],
                "val": [n for n in names if n[0] in scenes_val],
            }
        elif isinstance(split_arg, str):
            if (self.root / split_arg).exists():
                # Common split file.
                with (self.root / split_arg).open("r") as fp:
                    splits = json.load(fp)
            else:
                # Per-scene split file.
                splits = defaultdict(dict)
                for scene in self.cfg.scenes:
                    with (self.root / split_arg.format(scene=scene)).open("r") as fp:
                        scene_splits = json.load(fp)
                    for split_name in scene_splits:
                        splits[split_name][scene] = scene_splits[split_name]
            splits = {
                split_name: {scene: set(ids) for scene, ids in split.items()}
                for split_name, split in splits.items()
            }
            self.splits = {}
            for split_name, split in splits.items():
                self.splits[split_name] = [
                    (scene, *arg, name)
                    for scene, *arg, name in names
                    if scene in split and int(name.rsplit("_", 1)[0]) in split[scene]
                ]
        else:
            raise ValueError(split_arg)

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.splits[stage],
            self.data[stage],
            self.image_dirs,
            self.tile_managers,
            image_ext=".jpg",
        )

    def dataloader(
        self,
        stage: str,
        shuffle: bool = False,
        num_workers: int = None,
        sampler: Optional[torchdata.Sampler] = None,
    ):
        dataset = self.dataset(stage)
        cfg = self.cfg["loading"][stage]
        num_workers = cfg["num_workers"] if num_workers is None else num_workers
        loader = torchdata.DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=num_workers,
            shuffle=shuffle or (stage == "train"),
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)

    def sequence_dataset(self, stage: str, **kwargs):
        keys = self.splits[stage]
        seq2indices = defaultdict(list)
        for index, (_, seq, _) in enumerate(keys):
            seq2indices[seq].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(self.data[stage], indices, **kwargs)
            for i, sub_indices in enumerate(chunks):
                chunk2indices[seq, i] = sub_indices
        # store the index of each chunk in its sequence
        chunk_indices = torch.full((len(keys),), -1)
        for (_, chunk_index), idx in chunk2indices.items():
            chunk_indices[idx] = chunk_index
        self.data[stage]["chunk_index"] = chunk_indices
        dataset = self.dataset(stage)
        return dataset, chunk2indices

    def sequence_dataloader(self, stage: str, shuffle: bool = False, **kwargs):
        dataset, chunk2idx = self.sequence_dataset(stage, **kwargs)
        chunk_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(chunk_keys))
            chunk_keys = [chunk_keys[i] for i in perm]
        key_indices = [i for key in chunk_keys for i in chunk2idx[key]]
        num_workers = self.cfg["loading"][stage]["num_workers"]
        loader = torchdata.DataLoader(
            dataset,
            batch_size=None,
            sampler=key_indices,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return loader, chunk_keys, chunk2idx

class MapillaryRainyDataModule(pl.LightningDataModule):
    dump_filename = "dump.json"
    # images_archive = "images.tar.gz"
    images_dirname = "images/"

    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "mapillary_rainy",
        # paths and fetch
        "data_dir": DATASETS_PATH / "MGL_final",
        "noise_data_dir": DATASETS_PATH / "MGL_rainy",
        "local_dir": None,
        "tiles_filename": "tiles.pkl",
        "scenes": "???",
        "split": None,
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 1, "num_workers": 0},
        },
        "filter_for": None,
        "filter_by_ground_angle": None,
        "min_num_points": "???",
    }

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.noise_root = Path(self.cfg.noise_data_dir)
        self.local_dir = self.cfg.local_dir or os.environ.get("TMPDIR")
        if self.local_dir is not None:
            self.local_dir = Path(self.local_dir, "MGL_rainy")
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")

    def prepare_data(self):
        for scene in self.cfg.scenes:
            dump_dir = self.root / scene
            assert (dump_dir / self.dump_filename).exists(), dump_dir
            assert (dump_dir / self.cfg.tiles_filename).exists(), dump_dir
            if self.local_dir is None:
                assert (self.noise_root / scene / self.images_dirname).exists(), dump_dir
                continue
            # Cache the folder of images locally to speed up reading
            local_dir = self.local_dir / scene
            if local_dir.exists():
                shutil.rmtree(local_dir)
            local_dir.mkdir(exist_ok=True, parents=True)
            # images_archive = dump_dir / self.images_archive
            # logger.info("Extracting the image archive %s.", images_archive)
            # with tarfile.open(images_archive) as fp:
                # fp.extractall(local_dir)

    def setup(self, stage: Optional[str] = None):
        self.dumps = {}
        self.tile_managers = {}
        self.image_dirs = {}
        names = []

        for scene in self.cfg.scenes:
            logger.info("Loading scene %s.", scene)
            dump_dir = self.root / scene
            logger.info("Loading map tiles %s.", self.cfg.tiles_filename)
            self.tile_managers[scene] = TileManager.load(
                dump_dir / self.cfg.tiles_filename
            )
            groups = self.tile_managers[scene].groups
            if self.cfg.num_classes:  # check consistency
                if set(groups.keys()) != set(self.cfg.num_classes.keys()):
                    raise ValueError(
                        "Inconsistent groups: "
                        f"{groups.keys()} {self.cfg.num_classes.keys()}"
                    )
                for k in groups:
                    if len(groups[k]) != self.cfg.num_classes[k]:
                        raise ValueError(
                            f"{k}: {len(groups[k])} vs {self.cfg.num_classes[k]}"
                        )
            ppm = self.tile_managers[scene].ppm
            if ppm != self.cfg.pixel_per_meter:
                raise ValueError(
                    "The tile manager and the config/model have different ground "
                    f"resolutions: {ppm} vs {self.cfg.pixel_per_meter}"
                )

            logger.info("Loading dump json file %s.", self.dump_filename)
            with (dump_dir / self.dump_filename).open("r") as fp:
                self.dumps[scene] = pack_dump_dict(json.load(fp))
            for seq, per_seq in self.dumps[scene].items():
                for cam_id, cam_dict in per_seq["cameras"].items():
                    if cam_dict["model"] != "PINHOLE":
                        raise ValueError(
                            "Unsupported camera model: "
                            f"{cam_dict['model']} for {scene},{seq},{cam_id}"
                        )

            self.image_dirs[scene] = (
                # (self.local_dir or self.root) / scene / self.images_dirname
                self.noise_root / scene / self.images_dirname
            )
            assert self.image_dirs[scene].exists(), self.image_dirs[scene]

            for seq, data in self.dumps[scene].items():
                for name in data["views"]:
                    names.append((scene, seq, name))

        self.parse_splits(self.cfg.split, names)
        if self.cfg.filter_for is not None:
            self.filter_elements()
        self.pack_data()

    def pack_data(self):
        # We pack the data into compact tensors
        # that can be shared across processes without copy.
        exclude = {
            "compass_angle",
            "compass_accuracy",
            "gps_accuracy",
            "chunk_key",
            "panorama_offset",
        }
        cameras = {
            scene: {seq: per_seq["cameras"] for seq, per_seq in per_scene.items()}
            for scene, per_scene in self.dumps.items()
        }
        points = {
            scene: {
                seq: {
                    i: torch.from_numpy(p) for i, p in per_seq.get("points", {}).items()
                }
                for seq, per_seq in per_scene.items()
            }
            for scene, per_scene in self.dumps.items()
        }
        self.data = {}
        for stage, names in self.splits.items():
            view = self.dumps[names[0][0]][names[0][1]]["views"][names[0][2]]
            data = {k: [] for k in view.keys() - exclude}
            for scene, seq, name in names:
                for k in data:
                    data[k].append(self.dumps[scene][seq]["views"][name].get(k, None))
            for k in data:
                v = np.array(data[k])
                if np.issubdtype(v.dtype, np.integer) or np.issubdtype(
                    v.dtype, np.floating
                ):
                    v = torch.from_numpy(v)
                data[k] = v
            data["cameras"] = cameras
            data["points"] = points
            self.data[stage] = data
            self.splits[stage] = np.array(names)

    def filter_elements(self):
        for stage, names in self.splits.items():
            names_select = []
            for scene, seq, name in names:
                view = self.dumps[scene][seq]["views"][name]
                if self.cfg.filter_for == "ground_plane":
                    if not (1.0 <= view["height"] <= 3.0):
                        continue
                    planes = self.dumps[scene][seq].get("plane")
                    if planes is not None:
                        inliers = planes[str(view["chunk_id"])][-1]
                        if inliers < 10:
                            continue
                    if self.cfg.filter_by_ground_angle is not None:
                        plane = np.array(view["plane_params"])
                        normal = plane[:3] / np.linalg.norm(plane[:3])
                        angle = np.rad2deg(np.arccos(np.abs(normal[-1])))
                        if angle > self.cfg.filter_by_ground_angle:
                            continue
                elif self.cfg.filter_for == "pointcloud":
                    if len(view["observations"]) < self.cfg.min_num_points:
                        continue
                elif self.cfg.filter_for is not None:
                    raise ValueError(f"Unknown filtering: {self.cfg.filter_for}")
                names_select.append((scene, seq, name))
            logger.info(
                "%s: Keep %d/%d images after filtering for %s.",
                stage,
                len(names_select),
                len(names),
                self.cfg.filter_for,
            )
            self.splits[stage] = names_select

    def parse_splits(self, split_arg, names):
        if split_arg is None:
            self.splits = {
                "train": names,
                "val": names,
            }
        elif isinstance(split_arg, int):
            names = np.random.RandomState(self.cfg.seed).permutation(names).tolist()
            self.splits = {
                "train": names[split_arg:],
                "val": names[:split_arg],
            }
        elif isinstance(split_arg, DictConfig):
            scenes_val = set(split_arg.val)
            scenes_train = set(split_arg.train)
            assert len(scenes_val - set(self.cfg.scenes)) == 0
            assert len(scenes_train - set(self.cfg.scenes)) == 0
            self.splits = {
                "train": [n for n in names if n[0] in scenes_train],
                "val": [n for n in names if n[0] in scenes_val],
            }
        elif isinstance(split_arg, str):
            if (self.root / split_arg).exists():
                # Common split file.
                with (self.root / split_arg).open("r") as fp:
                    splits = json.load(fp)
            else:
                # Per-scene split file.
                splits = defaultdict(dict)
                for scene in self.cfg.scenes:
                    with (self.root / split_arg.format(scene=scene)).open("r") as fp:
                        scene_splits = json.load(fp)
                    for split_name in scene_splits:
                        splits[split_name][scene] = scene_splits[split_name]
            splits = {
                split_name: {scene: set(ids) for scene, ids in split.items()}
                for split_name, split in splits.items()
            }
            self.splits = {}
            for split_name, split in splits.items():
                self.splits[split_name] = [
                    (scene, *arg, name)
                    for scene, *arg, name in names
                    if scene in split and int(name.rsplit("_", 1)[0]) in split[scene]
                ]
        else:
            raise ValueError(split_arg)

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.splits[stage],
            self.data[stage],
            self.image_dirs,
            self.tile_managers,
            image_ext=".jpg",
        )

    def dataloader(
        self,
        stage: str,
        shuffle: bool = False,
        num_workers: int = None,
        sampler: Optional[torchdata.Sampler] = None,
    ):
        dataset = self.dataset(stage)
        cfg = self.cfg["loading"][stage]
        num_workers = cfg["num_workers"] if num_workers is None else num_workers
        loader = torchdata.DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=num_workers,
            shuffle=shuffle or (stage == "train"),
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)

    def sequence_dataset(self, stage: str, **kwargs):
        keys = self.splits[stage]
        seq2indices = defaultdict(list)
        for index, (_, seq, _) in enumerate(keys):
            seq2indices[seq].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(self.data[stage], indices, **kwargs)
            for i, sub_indices in enumerate(chunks):
                chunk2indices[seq, i] = sub_indices
        # store the index of each chunk in its sequence
        chunk_indices = torch.full((len(keys),), -1)
        for (_, chunk_index), idx in chunk2indices.items():
            chunk_indices[idx] = chunk_index
        self.data[stage]["chunk_index"] = chunk_indices
        dataset = self.dataset(stage)
        return dataset, chunk2indices

    def sequence_dataloader(self, stage: str, shuffle: bool = False, **kwargs):
        dataset, chunk2idx = self.sequence_dataset(stage, **kwargs)
        chunk_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(chunk_keys))
            chunk_keys = [chunk_keys[i] for i in perm]
        key_indices = [i for key in chunk_keys for i in chunk2idx[key]]
        num_workers = self.cfg["loading"][stage]["num_workers"]
        loader = torchdata.DataLoader(
            dataset,
            batch_size=None,
            sampler=key_indices,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return loader, chunk_keys, chunk2idx

class MapillarySnowyDataModule(pl.LightningDataModule):
    dump_filename = "dump.json"
    # images_archive = "images.tar.gz"
    images_dirname = "images/"

    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "mapillary_snowy",
        # paths and fetch
        "data_dir": DATASETS_PATH / "MGL_final",
        "noise_data_dir": DATASETS_PATH / "MGL_snowy",
        "local_dir": None,
        "tiles_filename": "tiles.pkl",
        "scenes": "???",
        "split": None,
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 1, "num_workers": 0},
        },
        "filter_for": None,
        "filter_by_ground_angle": None,
        "min_num_points": "???",
    }

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.noise_root = Path(self.cfg.noise_data_dir)
        self.local_dir = self.cfg.local_dir or os.environ.get("TMPDIR")
        if self.local_dir is not None:
            self.local_dir = Path(self.local_dir, "MGL_snowy")
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")

    def prepare_data(self):
        for scene in self.cfg.scenes:
            dump_dir = self.root / scene
            assert (dump_dir / self.dump_filename).exists(), dump_dir
            assert (dump_dir / self.cfg.tiles_filename).exists(), dump_dir
            if self.local_dir is None:
                assert (self.noise_root / scene / self.images_dirname).exists(), dump_dir
                continue
            # Cache the folder of images locally to speed up reading
            local_dir = self.local_dir / scene
            if local_dir.exists():
                shutil.rmtree(local_dir)
            local_dir.mkdir(exist_ok=True, parents=True)
            # images_archive = dump_dir / self.images_archive
            # logger.info("Extracting the image archive %s.", images_archive)
            # with tarfile.open(images_archive) as fp:
                # fp.extractall(local_dir)

    def setup(self, stage: Optional[str] = None):
        self.dumps = {}
        self.tile_managers = {}
        self.image_dirs = {}
        names = []

        for scene in self.cfg.scenes:
            logger.info("Loading scene %s.", scene)
            dump_dir = self.root / scene
            logger.info("Loading map tiles %s.", self.cfg.tiles_filename)
            self.tile_managers[scene] = TileManager.load(
                dump_dir / self.cfg.tiles_filename
            )
            groups = self.tile_managers[scene].groups
            if self.cfg.num_classes:  # check consistency
                if set(groups.keys()) != set(self.cfg.num_classes.keys()):
                    raise ValueError(
                        "Inconsistent groups: "
                        f"{groups.keys()} {self.cfg.num_classes.keys()}"
                    )
                for k in groups:
                    if len(groups[k]) != self.cfg.num_classes[k]:
                        raise ValueError(
                            f"{k}: {len(groups[k])} vs {self.cfg.num_classes[k]}"
                        )
            ppm = self.tile_managers[scene].ppm
            if ppm != self.cfg.pixel_per_meter:
                raise ValueError(
                    "The tile manager and the config/model have different ground "
                    f"resolutions: {ppm} vs {self.cfg.pixel_per_meter}"
                )

            logger.info("Loading dump json file %s.", self.dump_filename)
            with (dump_dir / self.dump_filename).open("r") as fp:
                self.dumps[scene] = pack_dump_dict(json.load(fp))
            for seq, per_seq in self.dumps[scene].items():
                for cam_id, cam_dict in per_seq["cameras"].items():
                    if cam_dict["model"] != "PINHOLE":
                        raise ValueError(
                            "Unsupported camera model: "
                            f"{cam_dict['model']} for {scene},{seq},{cam_id}"
                        )

            self.image_dirs[scene] = (
                # (self.local_dir or self.root) / scene / self.images_dirname
                self.noise_root / scene / self.images_dirname
            )
            assert self.image_dirs[scene].exists(), self.image_dirs[scene]

            for seq, data in self.dumps[scene].items():
                for name in data["views"]:
                    names.append((scene, seq, name))

        self.parse_splits(self.cfg.split, names)
        if self.cfg.filter_for is not None:
            self.filter_elements()
        self.pack_data()

    def pack_data(self):
        # We pack the data into compact tensors
        # that can be shared across processes without copy.
        exclude = {
            "compass_angle",
            "compass_accuracy",
            "gps_accuracy",
            "chunk_key",
            "panorama_offset",
        }
        cameras = {
            scene: {seq: per_seq["cameras"] for seq, per_seq in per_scene.items()}
            for scene, per_scene in self.dumps.items()
        }
        points = {
            scene: {
                seq: {
                    i: torch.from_numpy(p) for i, p in per_seq.get("points", {}).items()
                }
                for seq, per_seq in per_scene.items()
            }
            for scene, per_scene in self.dumps.items()
        }
        self.data = {}
        for stage, names in self.splits.items():
            view = self.dumps[names[0][0]][names[0][1]]["views"][names[0][2]]
            data = {k: [] for k in view.keys() - exclude}
            for scene, seq, name in names:
                for k in data:
                    data[k].append(self.dumps[scene][seq]["views"][name].get(k, None))
            for k in data:
                v = np.array(data[k])
                if np.issubdtype(v.dtype, np.integer) or np.issubdtype(
                    v.dtype, np.floating
                ):
                    v = torch.from_numpy(v)
                data[k] = v
            data["cameras"] = cameras
            data["points"] = points
            self.data[stage] = data
            self.splits[stage] = np.array(names)

    def filter_elements(self):
        for stage, names in self.splits.items():
            names_select = []
            for scene, seq, name in names:
                view = self.dumps[scene][seq]["views"][name]
                if self.cfg.filter_for == "ground_plane":
                    if not (1.0 <= view["height"] <= 3.0):
                        continue
                    planes = self.dumps[scene][seq].get("plane")
                    if planes is not None:
                        inliers = planes[str(view["chunk_id"])][-1]
                        if inliers < 10:
                            continue
                    if self.cfg.filter_by_ground_angle is not None:
                        plane = np.array(view["plane_params"])
                        normal = plane[:3] / np.linalg.norm(plane[:3])
                        angle = np.rad2deg(np.arccos(np.abs(normal[-1])))
                        if angle > self.cfg.filter_by_ground_angle:
                            continue
                elif self.cfg.filter_for == "pointcloud":
                    if len(view["observations"]) < self.cfg.min_num_points:
                        continue
                elif self.cfg.filter_for is not None:
                    raise ValueError(f"Unknown filtering: {self.cfg.filter_for}")
                names_select.append((scene, seq, name))
            logger.info(
                "%s: Keep %d/%d images after filtering for %s.",
                stage,
                len(names_select),
                len(names),
                self.cfg.filter_for,
            )
            self.splits[stage] = names_select

    def parse_splits(self, split_arg, names):
        if split_arg is None:
            self.splits = {
                "train": names,
                "val": names,
            }
        elif isinstance(split_arg, int):
            names = np.random.RandomState(self.cfg.seed).permutation(names).tolist()
            self.splits = {
                "train": names[split_arg:],
                "val": names[:split_arg],
            }
        elif isinstance(split_arg, DictConfig):
            scenes_val = set(split_arg.val)
            scenes_train = set(split_arg.train)
            assert len(scenes_val - set(self.cfg.scenes)) == 0
            assert len(scenes_train - set(self.cfg.scenes)) == 0
            self.splits = {
                "train": [n for n in names if n[0] in scenes_train],
                "val": [n for n in names if n[0] in scenes_val],
            }
        elif isinstance(split_arg, str):
            if (self.root / split_arg).exists():
                # Common split file.
                with (self.root / split_arg).open("r") as fp:
                    splits = json.load(fp)
            else:
                # Per-scene split file.
                splits = defaultdict(dict)
                for scene in self.cfg.scenes:
                    with (self.root / split_arg.format(scene=scene)).open("r") as fp:
                        scene_splits = json.load(fp)
                    for split_name in scene_splits:
                        splits[split_name][scene] = scene_splits[split_name]
            splits = {
                split_name: {scene: set(ids) for scene, ids in split.items()}
                for split_name, split in splits.items()
            }
            self.splits = {}
            for split_name, split in splits.items():
                self.splits[split_name] = [
                    (scene, *arg, name)
                    for scene, *arg, name in names
                    if scene in split and int(name.rsplit("_", 1)[0]) in split[scene]
                ]
        else:
            raise ValueError(split_arg)

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.splits[stage],
            self.data[stage],
            self.image_dirs,
            self.tile_managers,
            image_ext=".jpg",
        )

    def dataloader(
        self,
        stage: str,
        shuffle: bool = False,
        num_workers: int = None,
        sampler: Optional[torchdata.Sampler] = None,
    ):
        dataset = self.dataset(stage)
        cfg = self.cfg["loading"][stage]
        num_workers = cfg["num_workers"] if num_workers is None else num_workers
        loader = torchdata.DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=num_workers,
            shuffle=shuffle or (stage == "train"),
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)

    def sequence_dataset(self, stage: str, **kwargs):
        keys = self.splits[stage]
        seq2indices = defaultdict(list)
        for index, (_, seq, _) in enumerate(keys):
            seq2indices[seq].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(self.data[stage], indices, **kwargs)
            for i, sub_indices in enumerate(chunks):
                chunk2indices[seq, i] = sub_indices
        # store the index of each chunk in its sequence
        chunk_indices = torch.full((len(keys),), -1)
        for (_, chunk_index), idx in chunk2indices.items():
            chunk_indices[idx] = chunk_index
        self.data[stage]["chunk_index"] = chunk_indices
        dataset = self.dataset(stage)
        return dataset, chunk2indices

    def sequence_dataloader(self, stage: str, shuffle: bool = False, **kwargs):
        dataset, chunk2idx = self.sequence_dataset(stage, **kwargs)
        chunk_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(chunk_keys))
            chunk_keys = [chunk_keys[i] for i in perm]
        key_indices = [i for key in chunk_keys for i in chunk2idx[key]]
        num_workers = self.cfg["loading"][stage]["num_workers"]
        loader = torchdata.DataLoader(
            dataset,
            batch_size=None,
            sampler=key_indices,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return loader, chunk_keys, chunk2idx

class MapillaryNightDataModule(pl.LightningDataModule):
    dump_filename = "dump.json"
    # images_archive = "images.tar.gz"
    images_dirname = "images/"

    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "mapillary_night",
        # paths and fetch
        "data_dir": DATASETS_PATH / "MGL_final",
        "noise_data_dir": DATASETS_PATH / "MGL_night",
        "local_dir": None,
        "tiles_filename": "tiles.pkl",
        "scenes": "???",
        "split": None,
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 1, "num_workers": 0},
        },
        "filter_for": None,
        "filter_by_ground_angle": None,
        "min_num_points": "???",
    }

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.noise_root = Path(self.cfg.noise_data_dir)
        self.local_dir = self.cfg.local_dir or os.environ.get("TMPDIR")
        if self.local_dir is not None:
            self.local_dir = Path(self.local_dir, "MGL_night")
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")

    def prepare_data(self):
        for scene in self.cfg.scenes:
            dump_dir = self.root / scene
            assert (dump_dir / self.dump_filename).exists(), dump_dir
            assert (dump_dir / self.cfg.tiles_filename).exists(), dump_dir
            if self.local_dir is None:
                assert (self.noise_root / scene / self.images_dirname).exists(), dump_dir
                continue
            # Cache the folder of images locally to speed up reading
            local_dir = self.local_dir / scene
            if local_dir.exists():
                shutil.rmtree(local_dir)
            local_dir.mkdir(exist_ok=True, parents=True)
            # images_archive = dump_dir / self.images_archive
            # logger.info("Extracting the image archive %s.", images_archive)
            # with tarfile.open(images_archive) as fp:
                # fp.extractall(local_dir)

    def setup(self, stage: Optional[str] = None):
        self.dumps = {}
        self.tile_managers = {}
        self.image_dirs = {}
        names = []

        for scene in self.cfg.scenes:
            logger.info("Loading scene %s.", scene)
            dump_dir = self.root / scene
            logger.info("Loading map tiles %s.", self.cfg.tiles_filename)
            self.tile_managers[scene] = TileManager.load(
                dump_dir / self.cfg.tiles_filename
            )
            groups = self.tile_managers[scene].groups
            if self.cfg.num_classes:  # check consistency
                if set(groups.keys()) != set(self.cfg.num_classes.keys()):
                    raise ValueError(
                        "Inconsistent groups: "
                        f"{groups.keys()} {self.cfg.num_classes.keys()}"
                    )
                for k in groups:
                    if len(groups[k]) != self.cfg.num_classes[k]:
                        raise ValueError(
                            f"{k}: {len(groups[k])} vs {self.cfg.num_classes[k]}"
                        )
            ppm = self.tile_managers[scene].ppm
            if ppm != self.cfg.pixel_per_meter:
                raise ValueError(
                    "The tile manager and the config/model have different ground "
                    f"resolutions: {ppm} vs {self.cfg.pixel_per_meter}"
                )

            logger.info("Loading dump json file %s.", self.dump_filename)
            with (dump_dir / self.dump_filename).open("r") as fp:
                self.dumps[scene] = pack_dump_dict(json.load(fp))
            for seq, per_seq in self.dumps[scene].items():
                for cam_id, cam_dict in per_seq["cameras"].items():
                    if cam_dict["model"] != "PINHOLE":
                        raise ValueError(
                            "Unsupported camera model: "
                            f"{cam_dict['model']} for {scene},{seq},{cam_id}"
                        )

            self.image_dirs[scene] = (
                # (self.local_dir or self.root) / scene / self.images_dirname
                self.noise_root / scene / self.images_dirname
            )
            assert self.image_dirs[scene].exists(), self.image_dirs[scene]

            for seq, data in self.dumps[scene].items():
                for name in data["views"]:
                    names.append((scene, seq, name))

        self.parse_splits(self.cfg.split, names)
        if self.cfg.filter_for is not None:
            self.filter_elements()
        self.pack_data()

    def pack_data(self):
        # We pack the data into compact tensors
        # that can be shared across processes without copy.
        exclude = {
            "compass_angle",
            "compass_accuracy",
            "gps_accuracy",
            "chunk_key",
            "panorama_offset",
        }
        cameras = {
            scene: {seq: per_seq["cameras"] for seq, per_seq in per_scene.items()}
            for scene, per_scene in self.dumps.items()
        }
        points = {
            scene: {
                seq: {
                    i: torch.from_numpy(p) for i, p in per_seq.get("points", {}).items()
                }
                for seq, per_seq in per_scene.items()
            }
            for scene, per_scene in self.dumps.items()
        }
        self.data = {}
        for stage, names in self.splits.items():
            view = self.dumps[names[0][0]][names[0][1]]["views"][names[0][2]]
            data = {k: [] for k in view.keys() - exclude}
            for scene, seq, name in names:
                for k in data:
                    data[k].append(self.dumps[scene][seq]["views"][name].get(k, None))
            for k in data:
                v = np.array(data[k])
                if np.issubdtype(v.dtype, np.integer) or np.issubdtype(
                    v.dtype, np.floating
                ):
                    v = torch.from_numpy(v)
                data[k] = v
            data["cameras"] = cameras
            data["points"] = points
            self.data[stage] = data
            self.splits[stage] = np.array(names)

    def filter_elements(self):
        for stage, names in self.splits.items():
            names_select = []
            for scene, seq, name in names:
                view = self.dumps[scene][seq]["views"][name]
                if self.cfg.filter_for == "ground_plane":
                    if not (1.0 <= view["height"] <= 3.0):
                        continue
                    planes = self.dumps[scene][seq].get("plane")
                    if planes is not None:
                        inliers = planes[str(view["chunk_id"])][-1]
                        if inliers < 10:
                            continue
                    if self.cfg.filter_by_ground_angle is not None:
                        plane = np.array(view["plane_params"])
                        normal = plane[:3] / np.linalg.norm(plane[:3])
                        angle = np.rad2deg(np.arccos(np.abs(normal[-1])))
                        if angle > self.cfg.filter_by_ground_angle:
                            continue
                elif self.cfg.filter_for == "pointcloud":
                    if len(view["observations"]) < self.cfg.min_num_points:
                        continue
                elif self.cfg.filter_for is not None:
                    raise ValueError(f"Unknown filtering: {self.cfg.filter_for}")
                names_select.append((scene, seq, name))
            logger.info(
                "%s: Keep %d/%d images after filtering for %s.",
                stage,
                len(names_select),
                len(names),
                self.cfg.filter_for,
            )
            self.splits[stage] = names_select

    def parse_splits(self, split_arg, names):
        if split_arg is None:
            self.splits = {
                "train": names,
                "val": names,
            }
        elif isinstance(split_arg, int):
            names = np.random.RandomState(self.cfg.seed).permutation(names).tolist()
            self.splits = {
                "train": names[split_arg:],
                "val": names[:split_arg],
            }
        elif isinstance(split_arg, DictConfig):
            scenes_val = set(split_arg.val)
            scenes_train = set(split_arg.train)
            assert len(scenes_val - set(self.cfg.scenes)) == 0
            assert len(scenes_train - set(self.cfg.scenes)) == 0
            self.splits = {
                "train": [n for n in names if n[0] in scenes_train],
                "val": [n for n in names if n[0] in scenes_val],
            }
        elif isinstance(split_arg, str):
            if (self.root / split_arg).exists():
                # Common split file.
                with (self.root / split_arg).open("r") as fp:
                    splits = json.load(fp)
            else:
                # Per-scene split file.
                splits = defaultdict(dict)
                for scene in self.cfg.scenes:
                    with (self.root / split_arg.format(scene=scene)).open("r") as fp:
                        scene_splits = json.load(fp)
                    for split_name in scene_splits:
                        splits[split_name][scene] = scene_splits[split_name]
            splits = {
                split_name: {scene: set(ids) for scene, ids in split.items()}
                for split_name, split in splits.items()
            }
            self.splits = {}
            for split_name, split in splits.items():
                self.splits[split_name] = [
                    (scene, *arg, name)
                    for scene, *arg, name in names
                    if scene in split and int(name.rsplit("_", 1)[0]) in split[scene]
                ]
        else:
            raise ValueError(split_arg)

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.splits[stage],
            self.data[stage],
            self.image_dirs,
            self.tile_managers,
            image_ext=".jpg",
        )

    def dataloader(
        self,
        stage: str,
        shuffle: bool = False,
        num_workers: int = None,
        sampler: Optional[torchdata.Sampler] = None,
    ):
        dataset = self.dataset(stage)
        cfg = self.cfg["loading"][stage]
        num_workers = cfg["num_workers"] if num_workers is None else num_workers
        loader = torchdata.DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=num_workers,
            shuffle=shuffle or (stage == "train"),
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)

    def sequence_dataset(self, stage: str, **kwargs):
        keys = self.splits[stage]
        seq2indices = defaultdict(list)
        for index, (_, seq, _) in enumerate(keys):
            seq2indices[seq].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(self.data[stage], indices, **kwargs)
            for i, sub_indices in enumerate(chunks):
                chunk2indices[seq, i] = sub_indices
        # store the index of each chunk in its sequence
        chunk_indices = torch.full((len(keys),), -1)
        for (_, chunk_index), idx in chunk2indices.items():
            chunk_indices[idx] = chunk_index
        self.data[stage]["chunk_index"] = chunk_indices
        dataset = self.dataset(stage)
        return dataset, chunk2indices

    def sequence_dataloader(self, stage: str, shuffle: bool = False, **kwargs):
        dataset, chunk2idx = self.sequence_dataset(stage, **kwargs)
        chunk_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(chunk_keys))
            chunk_keys = [chunk_keys[i] for i in perm]
        key_indices = [i for key in chunk_keys for i in chunk2idx[key]]
        num_workers = self.cfg["loading"][stage]["num_workers"]
        loader = torchdata.DataLoader(
            dataset,
            batch_size=None,
            sampler=key_indices,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return loader, chunk_keys, chunk2idx

class MapillaryFoggyDataModule(pl.LightningDataModule):
    dump_filename = "dump.json"
    # images_archive = "images.tar.gz"
    images_dirname = "images/"

    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "mapillary_foggy",
        # paths and fetch
        "data_dir": DATASETS_PATH / "MGL_final",
        "noise_data_dir": DATASETS_PATH / "MGL_foggy",
        "local_dir": None,
        "tiles_filename": "tiles.pkl",
        "scenes": "???",
        "split": None,
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 1, "num_workers": 0},
        },
        "filter_for": None,
        "filter_by_ground_angle": None,
        "min_num_points": "???",
    }

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.noise_root = Path(self.cfg.noise_data_dir)
        self.local_dir = self.cfg.local_dir or os.environ.get("TMPDIR")
        if self.local_dir is not None:
            self.local_dir = Path(self.local_dir, "MGL_foggy")
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")

    def prepare_data(self):
        for scene in self.cfg.scenes:
            dump_dir = self.root / scene
            assert (dump_dir / self.dump_filename).exists(), dump_dir
            assert (dump_dir / self.cfg.tiles_filename).exists(), dump_dir
            if self.local_dir is None:
                assert (self.noise_root / scene / self.images_dirname).exists(), dump_dir
                continue
            # Cache the folder of images locally to speed up reading
            local_dir = self.local_dir / scene
            if local_dir.exists():
                shutil.rmtree(local_dir)
            local_dir.mkdir(exist_ok=True, parents=True)
            # images_archive = dump_dir / self.images_archive
            # logger.info("Extracting the image archive %s.", images_archive)
            # with tarfile.open(images_archive) as fp:
                # fp.extractall(local_dir)

    def setup(self, stage: Optional[str] = None):
        self.dumps = {}
        self.tile_managers = {}
        self.image_dirs = {}
        names = []

        for scene in self.cfg.scenes:
            logger.info("Loading scene %s.", scene)
            dump_dir = self.root / scene
            logger.info("Loading map tiles %s.", self.cfg.tiles_filename)
            self.tile_managers[scene] = TileManager.load(
                dump_dir / self.cfg.tiles_filename
            )
            groups = self.tile_managers[scene].groups
            if self.cfg.num_classes:  # check consistency
                if set(groups.keys()) != set(self.cfg.num_classes.keys()):
                    raise ValueError(
                        "Inconsistent groups: "
                        f"{groups.keys()} {self.cfg.num_classes.keys()}"
                    )
                for k in groups:
                    if len(groups[k]) != self.cfg.num_classes[k]:
                        raise ValueError(
                            f"{k}: {len(groups[k])} vs {self.cfg.num_classes[k]}"
                        )
            ppm = self.tile_managers[scene].ppm
            if ppm != self.cfg.pixel_per_meter:
                raise ValueError(
                    "The tile manager and the config/model have different ground "
                    f"resolutions: {ppm} vs {self.cfg.pixel_per_meter}"
                )

            logger.info("Loading dump json file %s.", self.dump_filename)
            with (dump_dir / self.dump_filename).open("r") as fp:
                self.dumps[scene] = pack_dump_dict(json.load(fp))
            for seq, per_seq in self.dumps[scene].items():
                for cam_id, cam_dict in per_seq["cameras"].items():
                    if cam_dict["model"] != "PINHOLE":
                        raise ValueError(
                            "Unsupported camera model: "
                            f"{cam_dict['model']} for {scene},{seq},{cam_id}"
                        )

            self.image_dirs[scene] = (
                # (self.local_dir or self.root) / scene / self.images_dirname
                self.noise_root / scene / self.images_dirname
            )
            assert self.image_dirs[scene].exists(), self.image_dirs[scene]

            for seq, data in self.dumps[scene].items():
                for name in data["views"]:
                    names.append((scene, seq, name))

        self.parse_splits(self.cfg.split, names)
        if self.cfg.filter_for is not None:
            self.filter_elements()
        self.pack_data()

    def pack_data(self):
        # We pack the data into compact tensors
        # that can be shared across processes without copy.
        exclude = {
            "compass_angle",
            "compass_accuracy",
            "gps_accuracy",
            "chunk_key",
            "panorama_offset",
        }
        cameras = {
            scene: {seq: per_seq["cameras"] for seq, per_seq in per_scene.items()}
            for scene, per_scene in self.dumps.items()
        }
        points = {
            scene: {
                seq: {
                    i: torch.from_numpy(p) for i, p in per_seq.get("points", {}).items()
                }
                for seq, per_seq in per_scene.items()
            }
            for scene, per_scene in self.dumps.items()
        }
        self.data = {}
        for stage, names in self.splits.items():
            view = self.dumps[names[0][0]][names[0][1]]["views"][names[0][2]]
            data = {k: [] for k in view.keys() - exclude}
            for scene, seq, name in names:
                for k in data:
                    data[k].append(self.dumps[scene][seq]["views"][name].get(k, None))
            for k in data:
                v = np.array(data[k])
                if np.issubdtype(v.dtype, np.integer) or np.issubdtype(
                    v.dtype, np.floating
                ):
                    v = torch.from_numpy(v)
                data[k] = v
            data["cameras"] = cameras
            data["points"] = points
            self.data[stage] = data
            self.splits[stage] = np.array(names)

    def filter_elements(self):
        for stage, names in self.splits.items():
            names_select = []
            for scene, seq, name in names:
                view = self.dumps[scene][seq]["views"][name]
                if self.cfg.filter_for == "ground_plane":
                    if not (1.0 <= view["height"] <= 3.0):
                        continue
                    planes = self.dumps[scene][seq].get("plane")
                    if planes is not None:
                        inliers = planes[str(view["chunk_id"])][-1]
                        if inliers < 10:
                            continue
                    if self.cfg.filter_by_ground_angle is not None:
                        plane = np.array(view["plane_params"])
                        normal = plane[:3] / np.linalg.norm(plane[:3])
                        angle = np.rad2deg(np.arccos(np.abs(normal[-1])))
                        if angle > self.cfg.filter_by_ground_angle:
                            continue
                elif self.cfg.filter_for == "pointcloud":
                    if len(view["observations"]) < self.cfg.min_num_points:
                        continue
                elif self.cfg.filter_for is not None:
                    raise ValueError(f"Unknown filtering: {self.cfg.filter_for}")
                names_select.append((scene, seq, name))
            logger.info(
                "%s: Keep %d/%d images after filtering for %s.",
                stage,
                len(names_select),
                len(names),
                self.cfg.filter_for,
            )
            self.splits[stage] = names_select

    def parse_splits(self, split_arg, names):
        if split_arg is None:
            self.splits = {
                "train": names,
                "val": names,
            }
        elif isinstance(split_arg, int):
            names = np.random.RandomState(self.cfg.seed).permutation(names).tolist()
            self.splits = {
                "train": names[split_arg:],
                "val": names[:split_arg],
            }
        elif isinstance(split_arg, DictConfig):
            scenes_val = set(split_arg.val)
            scenes_train = set(split_arg.train)
            assert len(scenes_val - set(self.cfg.scenes)) == 0
            assert len(scenes_train - set(self.cfg.scenes)) == 0
            self.splits = {
                "train": [n for n in names if n[0] in scenes_train],
                "val": [n for n in names if n[0] in scenes_val],
            }
        elif isinstance(split_arg, str):
            if (self.root / split_arg).exists():
                # Common split file.
                with (self.root / split_arg).open("r") as fp:
                    splits = json.load(fp)
            else:
                # Per-scene split file.
                splits = defaultdict(dict)
                for scene in self.cfg.scenes:
                    with (self.root / split_arg.format(scene=scene)).open("r") as fp:
                        scene_splits = json.load(fp)
                    for split_name in scene_splits:
                        splits[split_name][scene] = scene_splits[split_name]
            splits = {
                split_name: {scene: set(ids) for scene, ids in split.items()}
                for split_name, split in splits.items()
            }
            self.splits = {}
            for split_name, split in splits.items():
                self.splits[split_name] = [
                    (scene, *arg, name)
                    for scene, *arg, name in names
                    if scene in split and int(name.rsplit("_", 1)[0]) in split[scene]
                ]
        else:
            raise ValueError(split_arg)

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.splits[stage],
            self.data[stage],
            self.image_dirs,
            self.tile_managers,
            image_ext=".jpg",
        )

    def dataloader(
        self,
        stage: str,
        shuffle: bool = False,
        num_workers: int = None,
        sampler: Optional[torchdata.Sampler] = None,
    ):
        dataset = self.dataset(stage)
        cfg = self.cfg["loading"][stage]
        num_workers = cfg["num_workers"] if num_workers is None else num_workers
        loader = torchdata.DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=num_workers,
            shuffle=shuffle or (stage == "train"),
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)

    def sequence_dataset(self, stage: str, **kwargs):
        keys = self.splits[stage]
        seq2indices = defaultdict(list)
        for index, (_, seq, _) in enumerate(keys):
            seq2indices[seq].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(self.data[stage], indices, **kwargs)
            for i, sub_indices in enumerate(chunks):
                chunk2indices[seq, i] = sub_indices
        # store the index of each chunk in its sequence
        chunk_indices = torch.full((len(keys),), -1)
        for (_, chunk_index), idx in chunk2indices.items():
            chunk_indices[idx] = chunk_index
        self.data[stage]["chunk_index"] = chunk_indices
        dataset = self.dataset(stage)
        return dataset, chunk2indices

    def sequence_dataloader(self, stage: str, shuffle: bool = False, **kwargs):
        dataset, chunk2idx = self.sequence_dataset(stage, **kwargs)
        chunk_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(chunk_keys))
            chunk_keys = [chunk_keys[i] for i in perm]
        key_indices = [i for key in chunk_keys for i in chunk2idx[key]]
        num_workers = self.cfg["loading"][stage]["num_workers"]
        loader = torchdata.DataLoader(
            dataset,
            batch_size=None,
            sampler=key_indices,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return loader, chunk_keys, chunk2idx

class MapillaryOverExposureDataModule(pl.LightningDataModule):
    dump_filename = "dump.json"
    # images_archive = "images.tar.gz"
    images_dirname = "images/"

    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "mapillary_over_exposure",
        # paths and fetch
        "data_dir": DATASETS_PATH / "MGL_final",
        "noise_data_dir": DATASETS_PATH / "MGL_over_exposure",
        "local_dir": None,
        "tiles_filename": "tiles.pkl",
        "scenes": "???",
        "split": None,
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 1, "num_workers": 0},
        },
        "filter_for": None,
        "filter_by_ground_angle": None,
        "min_num_points": "???",
    }

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.noise_root = Path(self.cfg.noise_data_dir)
        self.local_dir = self.cfg.local_dir or os.environ.get("TMPDIR")
        if self.local_dir is not None:
            self.local_dir = Path(self.local_dir, "MGL_over_exposure")
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")

    def prepare_data(self):
        for scene in self.cfg.scenes:
            dump_dir = self.root / scene
            assert (dump_dir / self.dump_filename).exists(), dump_dir
            assert (dump_dir / self.cfg.tiles_filename).exists(), dump_dir
            if self.local_dir is None:
                assert (self.noise_root / scene / self.images_dirname).exists(), dump_dir
                continue
            # Cache the folder of images locally to speed up reading
            local_dir = self.local_dir / scene
            if local_dir.exists():
                shutil.rmtree(local_dir)
            local_dir.mkdir(exist_ok=True, parents=True)
            # images_archive = dump_dir / self.images_archive
            # logger.info("Extracting the image archive %s.", images_archive)
            # with tarfile.open(images_archive) as fp:
                # fp.extractall(local_dir)

    def setup(self, stage: Optional[str] = None):
        self.dumps = {}
        self.tile_managers = {}
        self.image_dirs = {}
        names = []

        for scene in self.cfg.scenes:
            logger.info("Loading scene %s.", scene)
            dump_dir = self.root / scene
            logger.info("Loading map tiles %s.", self.cfg.tiles_filename)
            self.tile_managers[scene] = TileManager.load(
                dump_dir / self.cfg.tiles_filename
            )
            groups = self.tile_managers[scene].groups
            if self.cfg.num_classes:  # check consistency
                if set(groups.keys()) != set(self.cfg.num_classes.keys()):
                    raise ValueError(
                        "Inconsistent groups: "
                        f"{groups.keys()} {self.cfg.num_classes.keys()}"
                    )
                for k in groups:
                    if len(groups[k]) != self.cfg.num_classes[k]:
                        raise ValueError(
                            f"{k}: {len(groups[k])} vs {self.cfg.num_classes[k]}"
                        )
            ppm = self.tile_managers[scene].ppm
            if ppm != self.cfg.pixel_per_meter:
                raise ValueError(
                    "The tile manager and the config/model have different ground "
                    f"resolutions: {ppm} vs {self.cfg.pixel_per_meter}"
                )

            logger.info("Loading dump json file %s.", self.dump_filename)
            with (dump_dir / self.dump_filename).open("r") as fp:
                self.dumps[scene] = pack_dump_dict(json.load(fp))
            for seq, per_seq in self.dumps[scene].items():
                for cam_id, cam_dict in per_seq["cameras"].items():
                    if cam_dict["model"] != "PINHOLE":
                        raise ValueError(
                            "Unsupported camera model: "
                            f"{cam_dict['model']} for {scene},{seq},{cam_id}"
                        )

            self.image_dirs[scene] = (
                # (self.local_dir or self.root) / scene / self.images_dirname
                self.noise_root / scene / self.images_dirname
            )
            assert self.image_dirs[scene].exists(), self.image_dirs[scene]

            for seq, data in self.dumps[scene].items():
                for name in data["views"]:
                    names.append((scene, seq, name))

        self.parse_splits(self.cfg.split, names)
        if self.cfg.filter_for is not None:
            self.filter_elements()
        self.pack_data()

    def pack_data(self):
        # We pack the data into compact tensors
        # that can be shared across processes without copy.
        exclude = {
            "compass_angle",
            "compass_accuracy",
            "gps_accuracy",
            "chunk_key",
            "panorama_offset",
        }
        cameras = {
            scene: {seq: per_seq["cameras"] for seq, per_seq in per_scene.items()}
            for scene, per_scene in self.dumps.items()
        }
        points = {
            scene: {
                seq: {
                    i: torch.from_numpy(p) for i, p in per_seq.get("points", {}).items()
                }
                for seq, per_seq in per_scene.items()
            }
            for scene, per_scene in self.dumps.items()
        }
        self.data = {}
        for stage, names in self.splits.items():
            view = self.dumps[names[0][0]][names[0][1]]["views"][names[0][2]]
            data = {k: [] for k in view.keys() - exclude}
            for scene, seq, name in names:
                for k in data:
                    data[k].append(self.dumps[scene][seq]["views"][name].get(k, None))
            for k in data:
                v = np.array(data[k])
                if np.issubdtype(v.dtype, np.integer) or np.issubdtype(
                    v.dtype, np.floating
                ):
                    v = torch.from_numpy(v)
                data[k] = v
            data["cameras"] = cameras
            data["points"] = points
            self.data[stage] = data
            self.splits[stage] = np.array(names)

    def filter_elements(self):
        for stage, names in self.splits.items():
            names_select = []
            for scene, seq, name in names:
                view = self.dumps[scene][seq]["views"][name]
                if self.cfg.filter_for == "ground_plane":
                    if not (1.0 <= view["height"] <= 3.0):
                        continue
                    planes = self.dumps[scene][seq].get("plane")
                    if planes is not None:
                        inliers = planes[str(view["chunk_id"])][-1]
                        if inliers < 10:
                            continue
                    if self.cfg.filter_by_ground_angle is not None:
                        plane = np.array(view["plane_params"])
                        normal = plane[:3] / np.linalg.norm(plane[:3])
                        angle = np.rad2deg(np.arccos(np.abs(normal[-1])))
                        if angle > self.cfg.filter_by_ground_angle:
                            continue
                elif self.cfg.filter_for == "pointcloud":
                    if len(view["observations"]) < self.cfg.min_num_points:
                        continue
                elif self.cfg.filter_for is not None:
                    raise ValueError(f"Unknown filtering: {self.cfg.filter_for}")
                names_select.append((scene, seq, name))
            logger.info(
                "%s: Keep %d/%d images after filtering for %s.",
                stage,
                len(names_select),
                len(names),
                self.cfg.filter_for,
            )
            self.splits[stage] = names_select

    def parse_splits(self, split_arg, names):
        if split_arg is None:
            self.splits = {
                "train": names,
                "val": names,
            }
        elif isinstance(split_arg, int):
            names = np.random.RandomState(self.cfg.seed).permutation(names).tolist()
            self.splits = {
                "train": names[split_arg:],
                "val": names[:split_arg],
            }
        elif isinstance(split_arg, DictConfig):
            scenes_val = set(split_arg.val)
            scenes_train = set(split_arg.train)
            assert len(scenes_val - set(self.cfg.scenes)) == 0
            assert len(scenes_train - set(self.cfg.scenes)) == 0
            self.splits = {
                "train": [n for n in names if n[0] in scenes_train],
                "val": [n for n in names if n[0] in scenes_val],
            }
        elif isinstance(split_arg, str):
            if (self.root / split_arg).exists():
                # Common split file.
                with (self.root / split_arg).open("r") as fp:
                    splits = json.load(fp)
            else:
                # Per-scene split file.
                splits = defaultdict(dict)
                for scene in self.cfg.scenes:
                    with (self.root / split_arg.format(scene=scene)).open("r") as fp:
                        scene_splits = json.load(fp)
                    for split_name in scene_splits:
                        splits[split_name][scene] = scene_splits[split_name]
            splits = {
                split_name: {scene: set(ids) for scene, ids in split.items()}
                for split_name, split in splits.items()
            }
            self.splits = {}
            for split_name, split in splits.items():
                self.splits[split_name] = [
                    (scene, *arg, name)
                    for scene, *arg, name in names
                    if scene in split and int(name.rsplit("_", 1)[0]) in split[scene]
                ]
        else:
            raise ValueError(split_arg)

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.splits[stage],
            self.data[stage],
            self.image_dirs,
            self.tile_managers,
            image_ext=".jpg",
        )

    def dataloader(
        self,
        stage: str,
        shuffle: bool = False,
        num_workers: int = None,
        sampler: Optional[torchdata.Sampler] = None,
    ):
        dataset = self.dataset(stage)
        cfg = self.cfg["loading"][stage]
        num_workers = cfg["num_workers"] if num_workers is None else num_workers
        loader = torchdata.DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=num_workers,
            shuffle=shuffle or (stage == "train"),
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)

    def sequence_dataset(self, stage: str, **kwargs):
        keys = self.splits[stage]
        seq2indices = defaultdict(list)
        for index, (_, seq, _) in enumerate(keys):
            seq2indices[seq].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(self.data[stage], indices, **kwargs)
            for i, sub_indices in enumerate(chunks):
                chunk2indices[seq, i] = sub_indices
        # store the index of each chunk in its sequence
        chunk_indices = torch.full((len(keys),), -1)
        for (_, chunk_index), idx in chunk2indices.items():
            chunk_indices[idx] = chunk_index
        self.data[stage]["chunk_index"] = chunk_indices
        dataset = self.dataset(stage)
        return dataset, chunk2indices

    def sequence_dataloader(self, stage: str, shuffle: bool = False, **kwargs):
        dataset, chunk2idx = self.sequence_dataset(stage, **kwargs)
        chunk_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(chunk_keys))
            chunk_keys = [chunk_keys[i] for i in perm]
        key_indices = [i for key in chunk_keys for i in chunk2idx[key]]
        num_workers = self.cfg["loading"][stage]["num_workers"]
        loader = torchdata.DataLoader(
            dataset,
            batch_size=None,
            sampler=key_indices,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return loader, chunk_keys, chunk2idx

class MapillaryUnderExposureDataModule(pl.LightningDataModule):
    dump_filename = "dump.json"
    # images_archive = "images.tar.gz"
    images_dirname = "images/"

    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "mapillary_over_exposure",
        # paths and fetch
        "data_dir": DATASETS_PATH / "MGL_final",
        "noise_data_dir": DATASETS_PATH / "MGL_under_exposure",
        "local_dir": None,
        "tiles_filename": "tiles.pkl",
        "scenes": "???",
        "split": None,
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 1, "num_workers": 0},
        },
        "filter_for": None,
        "filter_by_ground_angle": None,
        "min_num_points": "???",
    }

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.noise_root = Path(self.cfg.noise_data_dir)
        self.local_dir = self.cfg.local_dir or os.environ.get("TMPDIR")
        if self.local_dir is not None:
            self.local_dir = Path(self.local_dir, "MGL_under_exposure")
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")

    def prepare_data(self):
        for scene in self.cfg.scenes:
            dump_dir = self.root / scene
            assert (dump_dir / self.dump_filename).exists(), dump_dir
            assert (dump_dir / self.cfg.tiles_filename).exists(), dump_dir
            if self.local_dir is None:
                assert (self.noise_root / scene / self.images_dirname).exists(), dump_dir
                continue
            # Cache the folder of images locally to speed up reading
            local_dir = self.local_dir / scene
            if local_dir.exists():
                shutil.rmtree(local_dir)
            local_dir.mkdir(exist_ok=True, parents=True)
            # images_archive = dump_dir / self.images_archive
            # logger.info("Extracting the image archive %s.", images_archive)
            # with tarfile.open(images_archive) as fp:
                # fp.extractall(local_dir)

    def setup(self, stage: Optional[str] = None):
        self.dumps = {}
        self.tile_managers = {}
        self.image_dirs = {}
        names = []

        for scene in self.cfg.scenes:
            logger.info("Loading scene %s.", scene)
            dump_dir = self.root / scene
            logger.info("Loading map tiles %s.", self.cfg.tiles_filename)
            self.tile_managers[scene] = TileManager.load(
                dump_dir / self.cfg.tiles_filename
            )
            groups = self.tile_managers[scene].groups
            if self.cfg.num_classes:  # check consistency
                if set(groups.keys()) != set(self.cfg.num_classes.keys()):
                    raise ValueError(
                        "Inconsistent groups: "
                        f"{groups.keys()} {self.cfg.num_classes.keys()}"
                    )
                for k in groups:
                    if len(groups[k]) != self.cfg.num_classes[k]:
                        raise ValueError(
                            f"{k}: {len(groups[k])} vs {self.cfg.num_classes[k]}"
                        )
            ppm = self.tile_managers[scene].ppm
            if ppm != self.cfg.pixel_per_meter:
                raise ValueError(
                    "The tile manager and the config/model have different ground "
                    f"resolutions: {ppm} vs {self.cfg.pixel_per_meter}"
                )

            logger.info("Loading dump json file %s.", self.dump_filename)
            with (dump_dir / self.dump_filename).open("r") as fp:
                self.dumps[scene] = pack_dump_dict(json.load(fp))
            for seq, per_seq in self.dumps[scene].items():
                for cam_id, cam_dict in per_seq["cameras"].items():
                    if cam_dict["model"] != "PINHOLE":
                        raise ValueError(
                            "Unsupported camera model: "
                            f"{cam_dict['model']} for {scene},{seq},{cam_id}"
                        )

            self.image_dirs[scene] = (
                # (self.local_dir or self.root) / scene / self.images_dirname
                self.noise_root / scene / self.images_dirname
            )
            assert self.image_dirs[scene].exists(), self.image_dirs[scene]

            for seq, data in self.dumps[scene].items():
                for name in data["views"]:
                    names.append((scene, seq, name))

        self.parse_splits(self.cfg.split, names)
        if self.cfg.filter_for is not None:
            self.filter_elements()
        self.pack_data()

    def pack_data(self):
        # We pack the data into compact tensors
        # that can be shared across processes without copy.
        exclude = {
            "compass_angle",
            "compass_accuracy",
            "gps_accuracy",
            "chunk_key",
            "panorama_offset",
        }
        cameras = {
            scene: {seq: per_seq["cameras"] for seq, per_seq in per_scene.items()}
            for scene, per_scene in self.dumps.items()
        }
        points = {
            scene: {
                seq: {
                    i: torch.from_numpy(p) for i, p in per_seq.get("points", {}).items()
                }
                for seq, per_seq in per_scene.items()
            }
            for scene, per_scene in self.dumps.items()
        }
        self.data = {}
        for stage, names in self.splits.items():
            view = self.dumps[names[0][0]][names[0][1]]["views"][names[0][2]]
            data = {k: [] for k in view.keys() - exclude}
            for scene, seq, name in names:
                for k in data:
                    data[k].append(self.dumps[scene][seq]["views"][name].get(k, None))
            for k in data:
                v = np.array(data[k])
                if np.issubdtype(v.dtype, np.integer) or np.issubdtype(
                    v.dtype, np.floating
                ):
                    v = torch.from_numpy(v)
                data[k] = v
            data["cameras"] = cameras
            data["points"] = points
            self.data[stage] = data
            self.splits[stage] = np.array(names)

    def filter_elements(self):
        for stage, names in self.splits.items():
            names_select = []
            for scene, seq, name in names:
                view = self.dumps[scene][seq]["views"][name]
                if self.cfg.filter_for == "ground_plane":
                    if not (1.0 <= view["height"] <= 3.0):
                        continue
                    planes = self.dumps[scene][seq].get("plane")
                    if planes is not None:
                        inliers = planes[str(view["chunk_id"])][-1]
                        if inliers < 10:
                            continue
                    if self.cfg.filter_by_ground_angle is not None:
                        plane = np.array(view["plane_params"])
                        normal = plane[:3] / np.linalg.norm(plane[:3])
                        angle = np.rad2deg(np.arccos(np.abs(normal[-1])))
                        if angle > self.cfg.filter_by_ground_angle:
                            continue
                elif self.cfg.filter_for == "pointcloud":
                    if len(view["observations"]) < self.cfg.min_num_points:
                        continue
                elif self.cfg.filter_for is not None:
                    raise ValueError(f"Unknown filtering: {self.cfg.filter_for}")
                names_select.append((scene, seq, name))
            logger.info(
                "%s: Keep %d/%d images after filtering for %s.",
                stage,
                len(names_select),
                len(names),
                self.cfg.filter_for,
            )
            self.splits[stage] = names_select

    def parse_splits(self, split_arg, names):
        if split_arg is None:
            self.splits = {
                "train": names,
                "val": names,
            }
        elif isinstance(split_arg, int):
            names = np.random.RandomState(self.cfg.seed).permutation(names).tolist()
            self.splits = {
                "train": names[split_arg:],
                "val": names[:split_arg],
            }
        elif isinstance(split_arg, DictConfig):
            scenes_val = set(split_arg.val)
            scenes_train = set(split_arg.train)
            assert len(scenes_val - set(self.cfg.scenes)) == 0
            assert len(scenes_train - set(self.cfg.scenes)) == 0
            self.splits = {
                "train": [n for n in names if n[0] in scenes_train],
                "val": [n for n in names if n[0] in scenes_val],
            }
        elif isinstance(split_arg, str):
            if (self.root / split_arg).exists():
                # Common split file.
                with (self.root / split_arg).open("r") as fp:
                    splits = json.load(fp)
            else:
                # Per-scene split file.
                splits = defaultdict(dict)
                for scene in self.cfg.scenes:
                    with (self.root / split_arg.format(scene=scene)).open("r") as fp:
                        scene_splits = json.load(fp)
                    for split_name in scene_splits:
                        splits[split_name][scene] = scene_splits[split_name]
            splits = {
                split_name: {scene: set(ids) for scene, ids in split.items()}
                for split_name, split in splits.items()
            }
            self.splits = {}
            for split_name, split in splits.items():
                self.splits[split_name] = [
                    (scene, *arg, name)
                    for scene, *arg, name in names
                    if scene in split and int(name.rsplit("_", 1)[0]) in split[scene]
                ]
        else:
            raise ValueError(split_arg)

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.splits[stage],
            self.data[stage],
            self.image_dirs,
            self.tile_managers,
            image_ext=".jpg",
        )

    def dataloader(
        self,
        stage: str,
        shuffle: bool = False,
        num_workers: int = None,
        sampler: Optional[torchdata.Sampler] = None,
    ):
        dataset = self.dataset(stage)
        cfg = self.cfg["loading"][stage]
        num_workers = cfg["num_workers"] if num_workers is None else num_workers
        loader = torchdata.DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=num_workers,
            shuffle=shuffle or (stage == "train"),
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)

    def sequence_dataset(self, stage: str, **kwargs):
        keys = self.splits[stage]
        seq2indices = defaultdict(list)
        for index, (_, seq, _) in enumerate(keys):
            seq2indices[seq].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(self.data[stage], indices, **kwargs)
            for i, sub_indices in enumerate(chunks):
                chunk2indices[seq, i] = sub_indices
        # store the index of each chunk in its sequence
        chunk_indices = torch.full((len(keys),), -1)
        for (_, chunk_index), idx in chunk2indices.items():
            chunk_indices[idx] = chunk_index
        self.data[stage]["chunk_index"] = chunk_indices
        dataset = self.dataset(stage)
        return dataset, chunk2indices

    def sequence_dataloader(self, stage: str, shuffle: bool = False, **kwargs):
        dataset, chunk2idx = self.sequence_dataset(stage, **kwargs)
        chunk_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(chunk_keys))
            chunk_keys = [chunk_keys[i] for i in perm]
        key_indices = [i for key in chunk_keys for i in chunk2idx[key]]
        num_workers = self.cfg["loading"][stage]["num_workers"]
        loader = torchdata.DataLoader(
            dataset,
            batch_size=None,
            sampler=key_indices,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return loader, chunk_keys, chunk2idx

class MapillaryMotionBlurDataModule(pl.LightningDataModule):
    dump_filename = "dump.json"
    # images_archive = "images.tar.gz"
    images_dirname = "images/"

    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "mapillary_motion_blur",
        # paths and fetch
        "data_dir": DATASETS_PATH / "MGL_final",
        "noise_data_dir": DATASETS_PATH / "MGL_motion_blur",
        "local_dir": None,
        "tiles_filename": "tiles.pkl",
        "scenes": "???",
        "split": None,
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 1, "num_workers": 0},
        },
        "filter_for": None,
        "filter_by_ground_angle": None,
        "min_num_points": "???",
    }

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.noise_root = Path(self.cfg.noise_data_dir)
        self.local_dir = self.cfg.local_dir or os.environ.get("TMPDIR")
        if self.local_dir is not None:
            self.local_dir = Path(self.local_dir, "MGL_motion_blur")
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")

    def prepare_data(self):
        for scene in self.cfg.scenes:
            dump_dir = self.root / scene
            assert (dump_dir / self.dump_filename).exists(), dump_dir
            assert (dump_dir / self.cfg.tiles_filename).exists(), dump_dir
            if self.local_dir is None:
                assert (self.noise_root / scene / self.images_dirname).exists(), dump_dir
                continue
            # Cache the folder of images locally to speed up reading
            local_dir = self.local_dir / scene
            if local_dir.exists():
                shutil.rmtree(local_dir)
            local_dir.mkdir(exist_ok=True, parents=True)
            # images_archive = dump_dir / self.images_archive
            # logger.info("Extracting the image archive %s.", images_archive)
            # with tarfile.open(images_archive) as fp:
                # fp.extractall(local_dir)

    def setup(self, stage: Optional[str] = None):
        self.dumps = {}
        self.tile_managers = {}
        self.image_dirs = {}
        names = []

        for scene in self.cfg.scenes:
            logger.info("Loading scene %s.", scene)
            dump_dir = self.root / scene
            logger.info("Loading map tiles %s.", self.cfg.tiles_filename)
            self.tile_managers[scene] = TileManager.load(
                dump_dir / self.cfg.tiles_filename
            )
            groups = self.tile_managers[scene].groups
            if self.cfg.num_classes:  # check consistency
                if set(groups.keys()) != set(self.cfg.num_classes.keys()):
                    raise ValueError(
                        "Inconsistent groups: "
                        f"{groups.keys()} {self.cfg.num_classes.keys()}"
                    )
                for k in groups:
                    if len(groups[k]) != self.cfg.num_classes[k]:
                        raise ValueError(
                            f"{k}: {len(groups[k])} vs {self.cfg.num_classes[k]}"
                        )
            ppm = self.tile_managers[scene].ppm
            if ppm != self.cfg.pixel_per_meter:
                raise ValueError(
                    "The tile manager and the config/model have different ground "
                    f"resolutions: {ppm} vs {self.cfg.pixel_per_meter}"
                )

            logger.info("Loading dump json file %s.", self.dump_filename)
            with (dump_dir / self.dump_filename).open("r") as fp:
                self.dumps[scene] = pack_dump_dict(json.load(fp))
            for seq, per_seq in self.dumps[scene].items():
                for cam_id, cam_dict in per_seq["cameras"].items():
                    if cam_dict["model"] != "PINHOLE":
                        raise ValueError(
                            "Unsupported camera model: "
                            f"{cam_dict['model']} for {scene},{seq},{cam_id}"
                        )

            self.image_dirs[scene] = (
                # (self.local_dir or self.root) / scene / self.images_dirname
                self.noise_root / scene / self.images_dirname
            )
            assert self.image_dirs[scene].exists(), self.image_dirs[scene]

            for seq, data in self.dumps[scene].items():
                for name in data["views"]:
                    names.append((scene, seq, name))

        self.parse_splits(self.cfg.split, names)
        if self.cfg.filter_for is not None:
            self.filter_elements()
        self.pack_data()

    def pack_data(self):
        # We pack the data into compact tensors
        # that can be shared across processes without copy.
        exclude = {
            "compass_angle",
            "compass_accuracy",
            "gps_accuracy",
            "chunk_key",
            "panorama_offset",
        }
        cameras = {
            scene: {seq: per_seq["cameras"] for seq, per_seq in per_scene.items()}
            for scene, per_scene in self.dumps.items()
        }
        points = {
            scene: {
                seq: {
                    i: torch.from_numpy(p) for i, p in per_seq.get("points", {}).items()
                }
                for seq, per_seq in per_scene.items()
            }
            for scene, per_scene in self.dumps.items()
        }
        self.data = {}
        for stage, names in self.splits.items():
            view = self.dumps[names[0][0]][names[0][1]]["views"][names[0][2]]
            data = {k: [] for k in view.keys() - exclude}
            for scene, seq, name in names:
                for k in data:
                    data[k].append(self.dumps[scene][seq]["views"][name].get(k, None))
            for k in data:
                v = np.array(data[k])
                if np.issubdtype(v.dtype, np.integer) or np.issubdtype(
                    v.dtype, np.floating
                ):
                    v = torch.from_numpy(v)
                data[k] = v
            data["cameras"] = cameras
            data["points"] = points
            self.data[stage] = data
            self.splits[stage] = np.array(names)

    def filter_elements(self):
        for stage, names in self.splits.items():
            names_select = []
            for scene, seq, name in names:
                view = self.dumps[scene][seq]["views"][name]
                if self.cfg.filter_for == "ground_plane":
                    if not (1.0 <= view["height"] <= 3.0):
                        continue
                    planes = self.dumps[scene][seq].get("plane")
                    if planes is not None:
                        inliers = planes[str(view["chunk_id"])][-1]
                        if inliers < 10:
                            continue
                    if self.cfg.filter_by_ground_angle is not None:
                        plane = np.array(view["plane_params"])
                        normal = plane[:3] / np.linalg.norm(plane[:3])
                        angle = np.rad2deg(np.arccos(np.abs(normal[-1])))
                        if angle > self.cfg.filter_by_ground_angle:
                            continue
                elif self.cfg.filter_for == "pointcloud":
                    if len(view["observations"]) < self.cfg.min_num_points:
                        continue
                elif self.cfg.filter_for is not None:
                    raise ValueError(f"Unknown filtering: {self.cfg.filter_for}")
                names_select.append((scene, seq, name))
            logger.info(
                "%s: Keep %d/%d images after filtering for %s.",
                stage,
                len(names_select),
                len(names),
                self.cfg.filter_for,
            )
            self.splits[stage] = names_select

    def parse_splits(self, split_arg, names):
        if split_arg is None:
            self.splits = {
                "train": names,
                "val": names,
            }
        elif isinstance(split_arg, int):
            names = np.random.RandomState(self.cfg.seed).permutation(names).tolist()
            self.splits = {
                "train": names[split_arg:],
                "val": names[:split_arg],
            }
        elif isinstance(split_arg, DictConfig):
            scenes_val = set(split_arg.val)
            scenes_train = set(split_arg.train)
            assert len(scenes_val - set(self.cfg.scenes)) == 0
            assert len(scenes_train - set(self.cfg.scenes)) == 0
            self.splits = {
                "train": [n for n in names if n[0] in scenes_train],
                "val": [n for n in names if n[0] in scenes_val],
            }
        elif isinstance(split_arg, str):
            if (self.root / split_arg).exists():
                # Common split file.
                with (self.root / split_arg).open("r") as fp:
                    splits = json.load(fp)
            else:
                # Per-scene split file.
                splits = defaultdict(dict)
                for scene in self.cfg.scenes:
                    with (self.root / split_arg.format(scene=scene)).open("r") as fp:
                        scene_splits = json.load(fp)
                    for split_name in scene_splits:
                        splits[split_name][scene] = scene_splits[split_name]
            splits = {
                split_name: {scene: set(ids) for scene, ids in split.items()}
                for split_name, split in splits.items()
            }
            self.splits = {}
            for split_name, split in splits.items():
                self.splits[split_name] = [
                    (scene, *arg, name)
                    for scene, *arg, name in names
                    if scene in split and int(name.rsplit("_", 1)[0]) in split[scene]
                ]
        else:
            raise ValueError(split_arg)

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.splits[stage],
            self.data[stage],
            self.image_dirs,
            self.tile_managers,
            image_ext=".jpg",
        )

    def dataloader(
        self,
        stage: str,
        shuffle: bool = False,
        num_workers: int = None,
        sampler: Optional[torchdata.Sampler] = None,
    ):
        dataset = self.dataset(stage)
        cfg = self.cfg["loading"][stage]
        num_workers = cfg["num_workers"] if num_workers is None else num_workers
        loader = torchdata.DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=num_workers,
            shuffle=shuffle or (stage == "train"),
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)

    def sequence_dataset(self, stage: str, **kwargs):
        keys = self.splits[stage]
        seq2indices = defaultdict(list)
        for index, (_, seq, _) in enumerate(keys):
            seq2indices[seq].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(self.data[stage], indices, **kwargs)
            for i, sub_indices in enumerate(chunks):
                chunk2indices[seq, i] = sub_indices
        # store the index of each chunk in its sequence
        chunk_indices = torch.full((len(keys),), -1)
        for (_, chunk_index), idx in chunk2indices.items():
            chunk_indices[idx] = chunk_index
        self.data[stage]["chunk_index"] = chunk_indices
        dataset = self.dataset(stage)
        return dataset, chunk2indices

    def sequence_dataloader(self, stage: str, shuffle: bool = False, **kwargs):
        dataset, chunk2idx = self.sequence_dataset(stage, **kwargs)
        chunk_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(chunk_keys))
            chunk_keys = [chunk_keys[i] for i in perm]
        key_indices = [i for key in chunk_keys for i in chunk2idx[key]]
        num_workers = self.cfg["loading"][stage]["num_workers"]
        loader = torchdata.DataLoader(
            dataset,
            batch_size=None,
            sampler=key_indices,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return loader, chunk_keys, chunk2idx