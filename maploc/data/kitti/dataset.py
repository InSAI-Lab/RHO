# Copyright (c) Meta Platforms, Inc. and affiliates.

import collections
import collections.abc
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as torchdata
from torch.utils.data import Dataset, RandomSampler, SequentialSampler, DistributedSampler
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from ... import DATASETS_PATH, logger
from ...osm.tiling import TileManager
from ..dataset import MapLocDataset
from ..sampler import MinDistanceBatchSampler
from ..sequential import chunk_sequence
from ..torch import collate, worker_init_fn, contrastive_collate_fn
from .utils import get_camera_calibration, parse_gps_file, parse_split_file

class ContrastiveKittiDataset(Dataset):
    def __init__(self, anchor_dataset, positive_datasets, cfg):
        """
        A Dataset that pairs an anchor with its positive variations.

        Args:
            anchor_dataset (Dataset): The initialized dataset for clean anchor images.
            positive_datasets (dict[str, Dataset]): A dictionary of initialized datasets
                                                     for the noisy positive images.
        """
        super().__init__()
        self.anchor_dataset = anchor_dataset
        self.positive_datasets = positive_datasets
        self.cfg = cfg
        self.noise_types = list(self.positive_datasets.keys())

        # Sanity check: the number of noisy samples in each noisy dataset
        # should be the same to the number of samples in the anchor dataset.
        num_anchors = len(self.anchor_dataset)
        for noise, dataset in self.positive_datasets.items():
            assert len(dataset) == num_anchors, \
                f"Mismatch for '{noise}'. Anchor: {num_anchors}, Noisy: {len(dataset)}"

    def __len__(self):
        return len(self.anchor_dataset)

    def __getitem__(self, idx):
        """
        Fetches one anchor sample and all its corresponding positive images.
        """
        # 1. Get the complete data for the anchor sample
        anchor_data = self.anchor_dataset[idx]

        # 2. Get the corresponding positive images from each noisy dataset
        positive_images = []
        for noise in self.noise_types:
            positive_data = self.positive_datasets[noise][idx]
            positive_images.append(positive_data['image'])

        # 3. Add the list of positive images to the anchor's data dictionary
        anchor_data['positives'] = positive_images

        return anchor_data

class KittiContrastiveDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        # Make sure the config contains anchor and positives subconfig
        self.cfg = cfg
        self.anchor_dm = None
        self.positive_dms = {}
        # self.train_batch_sampler = None
        self.train_dataset = self.val_dataset = self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            stages = ['train', 'val']
        elif stage == None:
            stages = ['train', 'val', 'test']
        else:
            stages = [stage]
        
        # 1. Initialize the anchor DataModule
        self.anchor_dm = KittiDataModule(self.cfg.anchor)
        self.anchor_dm.setup(stage)

        # 2. Initialize all Positives DataModules
        noise_types = self.cfg.positives.keys()
        for noise in noise_types:
            if noise == "motion_blur":
                dm_class_name = "KittiMotionBlurDataModule"
            elif noise == "over_exposure":
                dm_class_name = "KittiOverExposureDataModule"
            elif noise == "under_exposure":
                dm_class_name = "KittiUnderExposureDataModule"
            else:
                dm_class_name = f"Kitti{noise.capitalize()}DataModule"
            dm_class = globals()[dm_class_name]
            noise_dm = dm_class(self.cfg.positives[noise])
            # Use the setyp method in the corresponding noise DataModule
            noise_dm.setup(stage)
            self.positive_dms[noise] = noise_dm
        
        # 3. Create the final train/val/test ContrastiveKittiDataset instances
        if 'train' in stages:
            positive_train_datasets = {n: dm.dataset('train') for n, dm in self.positive_dms.items()}
            self.train_dataset = ContrastiveKittiDataset(
                self.anchor_dm.dataset('train'), positive_train_datasets, self.cfg
            )
        
        if 'val' in stages:
            positive_val_datasets = {n: dm.dataset('val') for n, dm in self.positive_dms.items()}
            self.val_dataset = ContrastiveKittiDataset(
                self.anchor_dm.dataset('val'), positive_val_datasets, self.cfg
            )
        
        if 'test' in stages:
            positive_test_datasets = {n: dm.dataset('test') for n, dm in self.positive_dms.items()}
            self.test_dataset = ContrastiveKittiDataset(
                self.anchor_dm.dataset('test'), positive_test_datasets, self.cfg
            )
    
    def train_dataloader(self):
        if self.cfg.get('sampler') and self.cfg.sampler.name == 'min_distance':
            
            # Manually handle distributed sampling since use_distributed_sampler=False
            if torch.distributed.is_initialized():
                base_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            else:
                base_sampler = RandomSampler(self.train_dataset)

            positions = self.anchor_dm.data['train']['t_c2w'][:, :2]
            batch_sampler = MinDistanceBatchSampler(
                sampler=base_sampler,
                batch_size=self.cfg.loading.train.batch_size,
                drop_last=True,
                positions=positions,
                min_distance=self.cfg.sampler.min_distance
            )
            return torchdata.DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler, # Pass the fully constructed batch sampler
                num_workers=self.cfg.loading.train.num_workers,
                pin_memory=True,
                worker_init_fn=worker_init_fn,
                collate_fn=contrastive_collate_fn)

        else:
            # Default behavior without the custom sampler
            return torchdata.DataLoader(
                self.train_dataset,
                batch_size=self.cfg.loading.train.batch_size,
                shuffle=True, # Lightning will handle DDP for this simple case
                num_workers=self.cfg.loading.train.num_workers,
                pin_memory=True,
                worker_init_fn=worker_init_fn,
                collate_fn=contrastive_collate_fn)

    def val_dataloader(self):
        return torchdata.DataLoader(
            self.val_dataset,
            batch_size=self.cfg.loading.val.batch_size,
            num_workers=self.cfg.loading.val.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=self.cfg.loading.val.num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=contrastive_collate_fn
        )

    def test_dataloader(self):
        return torchdata.DataLoader(
            self.test_dataset,
            batch_size=self.cfg.loading.test.batch_size,
            num_workers=self.cfg.loading.test.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=self.cfg.loading.test.num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=contrastive_collate_fn
        )

class KittiMixDataModule(pl.LightningDataModule):
    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "kitti_mix",
        # paths and fetch
        "data_dir": DATASETS_PATH,
        "tiles_filename": "tiles.pkl",
        "splits": {
            "train": "train_files.txt",
            "val": "test1_files.txt",
            "test": "test2_files.txt",
        },
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 1, "num_workers": 0},
        },
        "max_num_val": 1000,
        "selection_subset_val": "furthest",
        "drop_train_too_close_to_val": 5.0,
        "skip_frames": 1,
        "camera_index": 2,
        # overwrite
        "crop_size_meters": 64,
        "max_init_error": 20,
        #"max_init_error_rotation": 10,
        "max_init_error_rotation_eval": 10,
        "add_map_mask": True,
        "mask_pad": 2,
        "target_focal_length": 256,
    }
    dummy_scene_name = "kitti_mix"

    def __init__(self, cfg, tile_manager: Optional[TileManager] = None):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.tile_manager = tile_manager
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")
        assert self.cfg.selection_subset_val in ["random", "furthest"]
        self.splits = {}
        self.shifts = {}
        self.calibrations = {}
        self.data = {}
        self.image_paths = {}

    def prepare_data(self):
        if not (self.root.exists() and (self.root / ".downloaded").exists()):
            raise FileNotFoundError(
                "Cannot find the KITTI dataset, run maploc.data.kitti.prepare"
            )

    def parse_split(self, split_arg):
        if isinstance(split_arg, str):
            names, shifts = parse_split_file(self.root / split_arg)
        elif isinstance(split_arg, collections.abc.Sequence):
            names = []
            shifts = None
            for date_drive in split_arg:
                data_dir = (
                    self.root / date_drive / f"image_{self.cfg.camera_index:02}/data"
                )
                assert data_dir.exists(), data_dir
                date_drive = tuple(date_drive.split("/"))
                n = sorted(date_drive + (p.name,) for p in data_dir.glob("*.png"))
                names.extend(n[:: self.cfg.skip_frames])
        else:
            raise ValueError(split_arg)
        return names, shifts

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            stages = ["train", "val"]
        elif stage is None:
            stages = ["train", "val", "test"]
        else:
            stages = [stage]
        for stage in stages:
            self.splits[stage], self.shifts[stage] = self.parse_split(
                self.cfg.splits[stage]
            )
        do_val_subset = "val" in stages and self.cfg.max_num_val is not None
        if do_val_subset and self.cfg.selection_subset_val == "random":
            select = np.random.RandomState(self.cfg.seed).choice(
                len(self.splits["val"]), self.cfg.max_num_val, replace=False
            )
            self.splits["val"] = [self.splits["val"][i] for i in select]
            if self.shifts["val"] is not None:
                self.shifts["val"] = self.shifts["val"][select]
        dates = {d for ns in self.splits.values() for _, _, d, _, _ in ns}
        for d in dates:
            self.calibrations[d] = get_camera_calibration(
                self.root / d, self.cfg.camera_index
            )
        if self.tile_manager is None:
            logger.info("Loading the tile manager...")
            self.tile_manager = TileManager.load(self.root / self.cfg.tiles_filename)
        self.cfg.num_classes = {k: len(g) for k, g in self.tile_manager.groups.items()}
        self.cfg.pixel_per_meter = self.tile_manager.ppm

        # pack all attributes in a single tensor to optimize memory access
        self.pack_data(stages)

        dists = None
        if do_val_subset and self.cfg.selection_subset_val == "furthest":
            dists = torch.cdist(
                self.data["val"]["t_c2w"][:, :2].double(),
                self.data["train"]["t_c2w"][:, :2].double(),
            )
            min_dists = dists.min(1).values
            select = torch.argsort(min_dists)[-self.cfg.max_num_val :]
            dists = dists[select]
            self.splits["val"] = [self.splits["val"][i] for i in select]
            if self.shifts["val"] is not None:
                self.shifts["val"] = self.shifts["val"][select]
            for k in list(self.data["val"]):
                if k != "cameras":
                    self.data["val"][k] = self.data["val"][k][select]
            self.image_paths["val"] = self.image_paths["val"][select]

        if "train" in stages and self.cfg.drop_train_too_close_to_val is not None:
            if dists is None:
                dists = torch.cdist(
                    self.data["val"]["t_c2w"][:, :2].double(),
                    self.data["train"]["t_c2w"][:, :2].double(),
                )
            drop = torch.any(dists < self.cfg.drop_train_too_close_to_val, 0)
            select = torch.where(~drop)[0]
            logger.info(
                "Dropping %d (%f %%) images that are too close to validation images.",
                drop.sum(),
                drop.float().mean(),
            )
            self.splits["train"] = [self.splits["train"][i] for i in select]
            if self.shifts["train"] is not None:
                self.shifts["train"] = self.shifts["train"][select]
            for k in list(self.data["train"]):
                if k != "cameras":
                    self.data["train"][k] = self.data["train"][k][select]
            self.image_paths["train"] = self.image_paths["train"][select]

    def pack_data(self, stages):
        for stage in stages:
            names = []
            data = {}
            for i, (noise_type, condition, date, drive, index) in enumerate(self.splits[stage]):
                d = self.get_frame_data(noise_type, condition, date, drive, index)
                for k, v in d.items():
                    if i == 0:
                        data[k] = []
                    data[k].append(v)
                path = f"{noise_type}/{condition}/{date}/{drive}/image_{self.cfg.camera_index:02}/data/{index}"
                names.append((self.dummy_scene_name, f"{noise_type}/{condition}/{date}/{drive}", path))
            for k in list(data):
                data[k] = torch.from_numpy(np.stack(data[k]))
            data["camera_id"] = np.full(len(names), self.cfg.camera_index)

            sequences = {noisetype_condition_date_drive for _, noisetype_condition_date_drive, _ in names}
            data["cameras"] = {
                self.dummy_scene_name: {
                    seq: {
                        self.cfg.camera_index: self.calibrations[seq.split("/")[2]][0]
                        # 获取的是对应日期（date）的、包含相机内参三个元素的完整元组 (K, R, t)
                        # [0] 索引从这个元组中取出了 第一个元素。
                    }
                    for seq in sequences
                }
            }
            shifts = self.shifts[stage]
            if shifts is not None:
                data["shifts"] = torch.from_numpy(shifts.astype(np.float32))
            self.data[stage] = data
            self.image_paths[stage] = np.array(names)

    def get_frame_data(self, noise_type, condition, date, drive, index):
        _, R_cam_gps, t_cam_gps = self.calibrations[date]

        # Transform the GPS pose to the camera pose
        gps_path = (
            self.root / noise_type / condition / date / drive / "oxts/data" / Path(index).with_suffix(".txt")
        )
        _, R_world_gps, t_world_gps = parse_gps_file(
            gps_path, self.tile_manager.projection
        )
        R_world_cam = R_world_gps @ R_cam_gps.T
        t_world_cam = t_world_gps - R_world_gps @ R_cam_gps.T @ t_cam_gps
        # Some voodoo to extract correct Euler angles from R_world_cam
        R_cv_xyz = Rotation.from_euler("YX", [-90, 90], degrees=True).as_matrix()
        R_world_cam_xyz = R_world_cam @ R_cv_xyz
        y, p, r = Rotation.from_matrix(R_world_cam_xyz).as_euler("ZYX", degrees=True)
        roll, pitch, yaw = r, -p, 90 - y
        roll_pitch_yaw = np.array([-roll, -pitch, yaw], np.float32)  # for some reason

        return {
            "t_c2w": t_world_cam.astype(np.float32),
            "roll_pitch_yaw": roll_pitch_yaw,
            "index": int(index.split(".")[0]),
        }

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.image_paths[stage],
            self.data[stage],
            {self.dummy_scene_name: self.root},
            {self.dummy_scene_name: self.tile_manager},
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
        keys = self.image_paths[stage]
        # group images by sequence (noise_type/condition/date/drive)
        seq2indices = defaultdict(list)
        for index, (_, noisetype_condition_date_drive, _) in enumerate(keys):
            seq2indices[noisetype_condition_date_drive].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(
                self.data[stage], indices, names=self.image_paths[stage], **kwargs
            )
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
        seq_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(seq_keys))
            seq_keys = [seq_keys[i] for i in perm]
        key_indices = [i for key in seq_keys for i in chunk2idx[key]]
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
        return loader, seq_keys, chunk2idx    

class KittiSplitDataModule(pl.LightningDataModule):
    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "kitti",
        # paths and fetch
        "data_dir": DATASETS_PATH / "kitti_split",
        "original_data_dir": DATASETS_PATH / "kitti",
        "tiles_filename": "tiles.pkl",
        "splits": {
            "train": "train_files_pano.txt",
            "val": "test1_files_pano.txt",
            "test": "test2_files_pano.txt",
        },
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 1, "num_workers": 0},
        },
        "max_num_val": 500,
        "selection_subset_val": "furthest",
        "drop_train_too_close_to_val": 5.0,
        "skip_frames": 1,
        "camera_index": 2,
        # overwrite
        "crop_size_meters": 64,
        "max_init_error": 20,
        #"max_init_error_rotation": 10,
        "max_init_error_rotation_eval": 10,
        "add_map_mask": True,
        "mask_pad": 2,
        "target_focal_length": 256,
    }
    dummy_scene_name = "kitti_split"

    def __init__(self, cfg, tile_manager: Optional[TileManager] = None):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.original_root = Path(self.cfg.original_data_dir)
        self.tile_manager = tile_manager
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")
        assert self.cfg.selection_subset_val in ["random", "furthest"]
        self.splits = {}
        self.shifts = {}
        self.calibrations = {}
        self.data = {}
        self.image_paths = {}
    '''
    def prepare_data(self):
        if not (self.root.exists() and (self.root / ".downloaded").exists()):
            raise FileNotFoundError(
                "Cannot find the KITTI dataset, run maploc.data.kitti.prepare"
            )
    '''
    def parse_split(self, split_arg):
        if isinstance(split_arg, str):
            names, shifts = parse_split_file(self.root / split_arg)
        elif isinstance(split_arg, collections.abc.Sequence):
            names = []
            shifts = None
            for date_drive in split_arg:
                data_dir = (
                    self.root / date_drive / f"image_{self.cfg.camera_index:02}/data"
                )
                assert data_dir.exists(), data_dir
                date_drive = tuple(date_drive.split("/"))
                n = sorted(date_drive + (p.name,) for p in data_dir.glob("*.png"))
                names.extend(n[:: self.cfg.skip_frames])
        else:
            raise ValueError(split_arg)
        return names, shifts

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            stages = ["train", "val"]
        elif stage is None:
            stages = ["train", "val", "test"]
        else:
            stages = [stage]
        for stage in stages:
            self.splits[stage], self.shifts[stage] = self.parse_split(
                self.cfg.splits[stage]
            )
        do_val_subset = "val" in stages and self.cfg.max_num_val is not None
        if do_val_subset and self.cfg.selection_subset_val == "random":
            select = np.random.RandomState(self.cfg.seed).choice(
                len(self.splits["val"]), self.cfg.max_num_val, replace=False
            )
            self.splits["val"] = [self.splits["val"][i] for i in select]
            if self.shifts["val"] is not None:
                self.shifts["val"] = self.shifts["val"][select]
        dates = {d for ns in self.splits.values() for d, _, _ in ns}
        for d in dates:
            self.calibrations[d] = get_camera_calibration(
                self.root / d, self.cfg.camera_index
            )
        if self.tile_manager is None:
            logger.info("Loading the tile manager...")
            self.tile_manager = TileManager.load(self.root / self.cfg.tiles_filename)
        self.cfg.num_classes = {k: len(g) for k, g in self.tile_manager.groups.items()}
        self.cfg.pixel_per_meter = self.tile_manager.ppm

        # pack all attributes in a single tensor to optimize memory access
        self.pack_data(stages)

        dists = None
        if do_val_subset and self.cfg.selection_subset_val == "furthest":
            dists = torch.cdist(
                self.data["val"]["t_c2w"][:, :2].double(),
                self.data["train"]["t_c2w"][:, :2].double(),
            )
            min_dists = dists.min(1).values
            select = torch.argsort(min_dists)[-self.cfg.max_num_val :]
            dists = dists[select]
            self.splits["val"] = [self.splits["val"][i] for i in select]
            if self.shifts["val"] is not None:
                self.shifts["val"] = self.shifts["val"][select]
            for k in list(self.data["val"]):
                if k != "cameras":
                    self.data["val"][k] = self.data["val"][k][select]
            self.image_paths["val"] = self.image_paths["val"][select]

        if "train" in stages and self.cfg.drop_train_too_close_to_val is not None:
            if dists is None:
                dists = torch.cdist(
                    self.data["val"]["t_c2w"][:, :2].double(),
                    self.data["train"]["t_c2w"][:, :2].double(),
                )
            drop = torch.any(dists < self.cfg.drop_train_too_close_to_val, 0)
            select = torch.where(~drop)[0]
            logger.info(
                "Dropping %d (%f %%) images that are too close to validation images.",
                drop.sum(),
                drop.float().mean(),
            )
            self.splits["train"] = [self.splits["train"][i] for i in select]
            if self.shifts["train"] is not None:
                self.shifts["train"] = self.shifts["train"][select]
            for k in list(self.data["train"]):
                if k != "cameras":
                    self.data["train"][k] = self.data["train"][k][select]
            self.image_paths["train"] = self.image_paths["train"][select]

    def pack_data(self, stages):
        for stage in stages:
            names = []
            data = {}
            for i, (date, drive, index) in enumerate(self.splits[stage]):
                d = self.get_frame_data(date, drive, index)
                for k, v in d.items():
                    if i == 0:
                        data[k] = []
                    data[k].append(v)
                path = f"{date}/{drive}/image_{self.cfg.camera_index:02}/data/{index}"
                names.append((self.dummy_scene_name, f"{date}/{drive}", path))
            for k in list(data):
                data[k] = torch.from_numpy(np.stack(data[k]))
            data["camera_id"] = np.full(len(names), self.cfg.camera_index)

            sequences = {date_drive for _, date_drive, _ in names}
            data["cameras"] = {
                self.dummy_scene_name: {
                    seq: {
                        self.cfg.camera_index: self.calibrations[seq.split("/")[0]][0]
                    }
                    for seq in sequences
                }
            }
            shifts = self.shifts[stage]
            if shifts is not None:
                data["shifts"] = torch.from_numpy(shifts.astype(np.float32))
            self.data[stage] = data
            self.image_paths[stage] = np.array(names)

    def get_frame_data(self, date, drive, index):
        _, R_cam_gps, t_cam_gps = self.calibrations[date]

        # Transform the GPS pose to the camera pose
        index_stem = index.split("_")[0]
        gps_path = (
            self.original_root / date / drive / "oxts/data" / Path(index_stem).with_suffix(".txt")
        )
        _, R_world_gps, t_world_gps = parse_gps_file(
            gps_path, self.tile_manager.projection
        )
        R_world_cam = R_world_gps @ R_cam_gps.T
        t_world_cam = t_world_gps - R_world_gps @ R_cam_gps.T @ t_cam_gps
        # Some voodoo to extract correct Euler angles from R_world_cam
        R_cv_xyz = Rotation.from_euler("YX", [-90, 90], degrees=True).as_matrix()
        R_world_cam_xyz = R_world_cam @ R_cv_xyz
        y, p, r = Rotation.from_matrix(R_world_cam_xyz).as_euler("ZYX", degrees=True)
        roll, pitch, yaw = r, -p, 90 - y
        roll_pitch_yaw = np.array([-roll, -pitch, yaw], np.float32)  # for some reason

        return {
            "t_c2w": t_world_cam.astype(np.float32),
            "roll_pitch_yaw": roll_pitch_yaw,
            "index": int(index_stem.split(".")[0]),
        }

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.image_paths[stage],
            self.data[stage],
            {self.dummy_scene_name: self.root},
            {self.dummy_scene_name: self.tile_manager},
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
        keys = self.image_paths[stage]
        # group images by sequence (date/drive)
        seq2indices = defaultdict(list)
        for index, (_, date_drive, _) in enumerate(keys):
            seq2indices[date_drive].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(
                self.data[stage], indices, names=self.image_paths[stage], **kwargs
            )
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
        seq_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(seq_keys))
            seq_keys = [seq_keys[i] for i in perm]
        key_indices = [i for key in seq_keys for i in chunk2idx[key]]
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
        return loader, seq_keys, chunk2idx

class KittiDataModule(pl.LightningDataModule):
    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "kitti",
        # paths and fetch
        "data_dir": DATASETS_PATH / "kitti",
        "tiles_filename": "tiles.pkl",
        "splits": {
            "train": "origin_train_files.txt",
            "val": "origin_test1_files.txt",
            "test": "origin_test2_files.txt",
        },
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 1, "num_workers": 0},
        },
        "max_num_val": 500,
        "selection_subset_val": "furthest",
        "drop_train_too_close_to_val": 5.0,
        "skip_frames": 1,
        "camera_index": 2,
        # overwrite
        "crop_size_meters": 64,
        "max_init_error": 20,
        #"max_init_error_rotation": 10,
        "max_init_error_rotation_eval": 10,
        "add_map_mask": True,
        "mask_pad": 2,
        "target_focal_length": 256,
    }
    dummy_scene_name = "kitti"

    def __init__(self, cfg, tile_manager: Optional[TileManager] = None):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.tile_manager = tile_manager
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")
        assert self.cfg.selection_subset_val in ["random", "furthest"]
        self.splits = {}
        self.shifts = {}
        self.calibrations = {}
        self.data = {}
        self.image_paths = {}
    '''
    def prepare_data(self):
        if not (self.root.exists() and (self.root / ".downloaded").exists()):
            raise FileNotFoundError(
                "Cannot find the KITTI dataset, run maploc.data.kitti.prepare"
            )
    '''
    def parse_split(self, split_arg):
        if isinstance(split_arg, str):
            names, shifts = parse_split_file(self.root / split_arg)
        elif isinstance(split_arg, collections.abc.Sequence):
            names = []
            shifts = None
            for date_drive in split_arg:
                data_dir = (
                    self.root / date_drive / f"image_{self.cfg.camera_index:02}/data"
                )
                assert data_dir.exists(), data_dir
                date_drive = tuple(date_drive.split("/"))
                n = sorted(date_drive + (p.name,) for p in data_dir.glob("*.png"))
                names.extend(n[:: self.cfg.skip_frames])
        else:
            raise ValueError(split_arg)
        return names, shifts

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            stages = ["train", "val"]
        elif stage is None:
            stages = ["train", "val", "test"]
        else:
            stages = [stage]
        for stage in stages:
            self.splits[stage], self.shifts[stage] = self.parse_split(
                self.cfg.splits[stage]
            )
        do_val_subset = "val" in stages and self.cfg.max_num_val is not None
        if do_val_subset and self.cfg.selection_subset_val == "random":
            select = np.random.RandomState(self.cfg.seed).choice(
                len(self.splits["val"]), self.cfg.max_num_val, replace=False
            )
            self.splits["val"] = [self.splits["val"][i] for i in select]
            if self.shifts["val"] is not None:
                self.shifts["val"] = self.shifts["val"][select]
        dates = {d for ns in self.splits.values() for d, _, _ in ns}
        for d in dates:
            self.calibrations[d] = get_camera_calibration(
                self.root / d, self.cfg.camera_index
            )
        if self.tile_manager is None:
            logger.info("Loading the tile manager...")
            self.tile_manager = TileManager.load(self.root / self.cfg.tiles_filename)
        self.cfg.num_classes = {k: len(g) for k, g in self.tile_manager.groups.items()}
        self.cfg.pixel_per_meter = self.tile_manager.ppm

        # pack all attributes in a single tensor to optimize memory access
        self.pack_data(stages)

        dists = None
        if do_val_subset and self.cfg.selection_subset_val == "furthest":
            dists = torch.cdist(
                self.data["val"]["t_c2w"][:, :2].double(),
                self.data["train"]["t_c2w"][:, :2].double(),
            )
            min_dists = dists.min(1).values
            select = torch.argsort(min_dists)[-self.cfg.max_num_val :]
            dists = dists[select]
            self.splits["val"] = [self.splits["val"][i] for i in select]
            if self.shifts["val"] is not None:
                self.shifts["val"] = self.shifts["val"][select]
            for k in list(self.data["val"]):
                if k != "cameras":
                    self.data["val"][k] = self.data["val"][k][select]
            self.image_paths["val"] = self.image_paths["val"][select]

        if "train" in stages and self.cfg.drop_train_too_close_to_val is not None:
            if dists is None:
                dists = torch.cdist(
                    self.data["val"]["t_c2w"][:, :2].double(),
                    self.data["train"]["t_c2w"][:, :2].double(),
                )
            drop = torch.any(dists < self.cfg.drop_train_too_close_to_val, 0)
            select = torch.where(~drop)[0]
            logger.info(
                "Dropping %d (%f %%) images that are too close to validation images.",
                drop.sum(),
                drop.float().mean(),
            )
            self.splits["train"] = [self.splits["train"][i] for i in select]
            if self.shifts["train"] is not None:
                self.shifts["train"] = self.shifts["train"][select]
            for k in list(self.data["train"]):
                if k != "cameras":
                    self.data["train"][k] = self.data["train"][k][select]
            self.image_paths["train"] = self.image_paths["train"][select]

    def pack_data(self, stages):
        for stage in stages:
            names = []
            data = {}
            for i, (date, drive, index) in enumerate(self.splits[stage]):
                d = self.get_frame_data(date, drive, index)
                for k, v in d.items():
                    if i == 0:
                        data[k] = []
                    data[k].append(v)
                path = f"{date}/{drive}/image_{self.cfg.camera_index:02}/data/{index}"
                names.append((self.dummy_scene_name, f"{date}/{drive}", path))
            for k in list(data):
                data[k] = torch.from_numpy(np.stack(data[k]))
            data["camera_id"] = np.full(len(names), self.cfg.camera_index)

            sequences = {date_drive for _, date_drive, _ in names}
            data["cameras"] = {
                self.dummy_scene_name: {
                    seq: {
                        self.cfg.camera_index: self.calibrations[seq.split("/")[0]][0]
                    }
                    for seq in sequences
                }
            }
            shifts = self.shifts[stage]
            if shifts is not None:
                data["shifts"] = torch.from_numpy(shifts.astype(np.float32))
            self.data[stage] = data
            self.image_paths[stage] = np.array(names)

    def get_frame_data(self, date, drive, index):
        _, R_cam_gps, t_cam_gps = self.calibrations[date]

        # Transform the GPS pose to the camera pose
        gps_path = (
            self.root / date / drive / "oxts/data" / Path(index).with_suffix(".txt")
        )
        _, R_world_gps, t_world_gps = parse_gps_file(
            gps_path, self.tile_manager.projection
        )
        R_world_cam = R_world_gps @ R_cam_gps.T
        t_world_cam = t_world_gps - R_world_gps @ R_cam_gps.T @ t_cam_gps
        # Some voodoo to extract correct Euler angles from R_world_cam
        R_cv_xyz = Rotation.from_euler("YX", [-90, 90], degrees=True).as_matrix()
        R_world_cam_xyz = R_world_cam @ R_cv_xyz
        y, p, r = Rotation.from_matrix(R_world_cam_xyz).as_euler("ZYX", degrees=True)
        roll, pitch, yaw = r, -p, 90 - y
        roll_pitch_yaw = np.array([-roll, -pitch, yaw], np.float32)  # for some reason

        return {
            "t_c2w": t_world_cam.astype(np.float32),
            "roll_pitch_yaw": roll_pitch_yaw,
            "index": int(index.split(".")[0]),
        }

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.image_paths[stage],
            self.data[stage],
            {self.dummy_scene_name: self.root},
            {self.dummy_scene_name: self.tile_manager},
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
        keys = self.image_paths[stage]
        # group images by sequence (date/drive)
        seq2indices = defaultdict(list)
        for index, (_, date_drive, _) in enumerate(keys):
            seq2indices[date_drive].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(
                self.data[stage], indices, names=self.image_paths[stage], **kwargs
            )
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
        seq_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(seq_keys))
            seq_keys = [seq_keys[i] for i in perm]
        key_indices = [i for key in seq_keys for i in chunk2idx[key]]
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
        return loader, seq_keys, chunk2idx

class KittiPanoDataModule(pl.LightningDataModule):
    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "kitti",
        # paths and fetch
        "data_dir": DATASETS_PATH / "kitti_split",
        "original_data_dir": DATASETS_PATH / "kitti",
        "tiles_filename": "tiles.pkl",
        "splits": {
            "train": "train_files_pano.txt",
            "val": "test1_files_pano.txt",
            "test": "test2_files_pano.txt",
        },
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 3, "num_workers": 0},
        },
        "max_num_val": 500,
        "selection_subset_val": "furthest",
        "drop_train_too_close_to_val": 5.0,
        "skip_frames": 1,
        "camera_index": 2,
        # overwrite
        "crop_size_meters": 64,
        "max_init_error": 20,
        #"max_init_error_rotation": 10,
        "max_init_error_rotation_eval": 10,
        "add_map_mask": True,
        "mask_pad": 2,
        "target_focal_length": 256,
        "group_pano_views": True,
    }
    dummy_scene_name = "kitti_split"

    def __init__(self, cfg, tile_manager: Optional[TileManager] = None):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.original_root = Path(self.cfg.original_data_dir)
        self.tile_manager = tile_manager
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")
        assert self.cfg.selection_subset_val in ["random", "furthest"]
        self.splits = {}
        self.shifts = {}
        self.calibrations = {}
        self.data = {}
        self.image_paths = {}
        self._is_setup = False
    '''
    def prepare_data(self):
        if not (self.root.exists() and (self.root / ".downloaded").exists()):
            raise FileNotFoundError(
                "Cannot find the KITTI dataset, run maploc.data.kitti.prepare"
            )
    '''
    def _group_views_by_panorama(self, names):
        pano_groups = defaultdict(list)
        for name in names:
            # name is a tuple: (date, drive, index_file)
            # e.g. ('2011_09_26', '2011_09_26_drive_0005_sync', '0000000048_left.png')
            index_file = name[-1]
            pano_id = f"{name[0]}_{name[1]}_{Path(index_file).stem.rsplit('_', 1)[0]}"
            pano_groups[pano_id].append(name)

        grouped_names = []
        for pano_id in sorted(pano_groups.keys()):
            views = sorted(pano_groups[pano_id], key=lambda n: n[-1])
            if len(views) != 3:
                logger.warning(f"Panorama {pano_id} has {len(views)} views, expected 3.")
                # continue
            grouped_names.extend(views)
        return grouped_names

    def parse_split(self, split_arg):
        if isinstance(split_arg, str):
            names, shifts = parse_split_file(self.root / split_arg)
        elif isinstance(split_arg, collections.abc.Sequence):
            names = []
            shifts = None
            for date_drive in split_arg:
                data_dir = (
                    self.root / date_drive / f"image_{self.cfg.camera_index:02}/data"
                )
                assert data_dir.exists(), data_dir
                date_drive = tuple(date_drive.split("/"))
                n = sorted(date_drive + (p.name,) for p in data_dir.glob("*.png"))
                names.extend(n[:: self.cfg.skip_frames])
        else:
            raise ValueError(split_arg)
        return names, shifts

    def setup(self, stage: Optional[str] = None):
        if self._is_setup:
            return
        
        if stage == "fit":
            stages = ["train", "val"]
        elif stage is None:
            stages = ["train", "val", "test"]
        else:
            stages = [stage]
        for stage_ in stages:
            self.splits[stage_], self.shifts[stage_] = self.parse_split(
                self.cfg.splits[stage_]
            )

        if self.cfg.group_pano_views:
            for stage_ in stages:
                self.splits[stage_] = self._group_views_by_panorama(self.splits[stage_])

        do_val_subset = "val" in stages and self.cfg.max_num_val is not None
        if do_val_subset and self.cfg.selection_subset_val == "random":
            select = np.random.RandomState(self.cfg.seed).choice(
                len(self.splits["val"]), self.cfg.max_num_val, replace=False
            )
            self.splits["val"] = [self.splits["val"][i] for i in select]
            if self.shifts["val"] is not None:
                self.shifts["val"] = self.shifts["val"][select]
        dates = {d for ns in self.splits.values() for d, _, _ in ns}
        for d in dates:
            self.calibrations[d] = get_camera_calibration(
                self.original_root / d, self.cfg.camera_index
            )
        if self.tile_manager is None:
            logger.info("Loading the tile manager...")
            self.tile_manager = TileManager.load(self.root / self.cfg.tiles_filename)
        self.cfg.num_classes = {k: len(g) for k, g in self.tile_manager.groups.items()}
        self.cfg.pixel_per_meter = self.tile_manager.ppm

        # pack all attributes in a single tensor to optimize memory access
        self.pack_data(stages)

        dists = None
        if do_val_subset and self.cfg.selection_subset_val == "furthest":
            num_panos_val = len(self.splits["val"]) // 3
            num_panos_train = len(self.splits["train"]) // 3
            # Get the position of the second view of each panorama
            pos_val = self.data["val"]["t_c2w"][1 : 3 * num_panos_val : 3, :2].double()
            pos_train = self.data["train"]["t_c2w"][1 : 3 * num_panos_train : 3, :2].double()

            dists = torch.cdist(pos_val, pos_train)

            min_dists = dists.min(1).values
            max_num_val_panos = self.cfg.max_num_val // 3
            select_panos = torch.argsort(min_dists)[-max_num_val_panos:]
            
            # Convert pano indices to item indices (3 views per pano)
            select = torch.cat([select_panos * 3 + i for i in range(3)]).sort().values

            dists = dists[select_panos]
            self.splits["val"] = [self.splits["val"][i] for i in select]
            if self.shifts["val"] is not None:
                self.shifts["val"] = self.shifts["val"][select]
            for k in list(self.data["val"]):
                if k != "cameras":
                    self.data["val"][k] = self.data["val"][k][select]
            self.image_paths["val"] = self.image_paths["val"][select]

        if "train" in stages and self.cfg.drop_train_too_close_to_val is not None:
            if dists is None:
                num_panos_val = len(self.splits["val"]) // 3
                num_panos_train = len(self.splits["train"]) // 3
                pos_val = self.data["val"]["t_c2w"][: 3 * num_panos_val : 3, :2].double()
                pos_train = self.data["train"]["t_c2w"][: 3 * num_panos_train : 3, :2].double()
                dists = torch.cdist(pos_val, pos_train)

            drop_panos = torch.any(dists < self.cfg.drop_train_too_close_to_val, 0)
            select_panos = torch.where(~drop_panos)[0]
            
            select = torch.cat([select_panos * 3 + i for i in range(3)]).sort().values

            logger.info(
                "Dropping %d (%f %%) images that are too close to validation images.",
                drop_panos.sum() * 3,
                drop_panos.float().mean(),
            )
            self.splits["train"] = [self.splits["train"][i] for i in select]
            if self.shifts["train"] is not None:
                self.shifts["train"] = self.shifts["train"][select]
            for k in list(self.data["train"]):
                if k != "cameras":
                    self.data["train"][k] = self.data["train"][k][select]
            self.image_paths["train"] = self.image_paths["train"][select]
        
        self._is_setup = True

    def pack_data(self, stages):
        for stage in stages:
            names = []
            data = {}
            for i, (date, drive, index) in enumerate(self.splits[stage]):
                d = self.get_frame_data(date, drive, index)
                for k, v in d.items():
                    if i == 0:
                        data[k] = []
                    data[k].append(v)
                path = f"{date}/{drive}/image_{self.cfg.camera_index:02}/data/{index}"
                names.append((self.dummy_scene_name, f"{date}/{drive}", path))
            for k in list(data):
                data[k] = torch.from_numpy(np.stack(data[k]))
            data["camera_id"] = np.full(len(names), self.cfg.camera_index)

            sequences = {date_drive for _, date_drive, _ in names}
            data["cameras"] = {
                self.dummy_scene_name: {
                    seq: {
                        self.cfg.camera_index: self.calibrations[seq.split("/")[0]][0]
                    }
                    for seq in sequences
                }
            }
            shifts = self.shifts[stage]
            if shifts is not None:
                data["shifts"] = torch.from_numpy(shifts.astype(np.float32))
            self.data[stage] = data
            self.image_paths[stage] = np.array(names)

    def get_frame_data(self, date, drive, index):
        _, R_cam_gps, t_cam_gps = self.calibrations[date]

        # Transform the GPS pose to the camera pose
        index_stem = Path(index).stem.split("_")[0]
        gps_path = (
            self.original_root / date / drive / "oxts/data" / Path(index_stem).with_suffix(".txt")
        )
        _, R_world_gps, t_world_gps = parse_gps_file(
            gps_path, self.tile_manager.projection
        )
        R_world_cam = R_world_gps @ R_cam_gps.T
        t_world_cam = t_world_gps - R_world_gps @ R_cam_gps.T @ t_cam_gps
        # Some voodoo to extract correct Euler angles from R_world_cam
        R_cv_xyz = Rotation.from_euler("YX", [-90, 90], degrees=True).as_matrix()
        R_world_cam_xyz = R_world_cam @ R_cv_xyz
        y, p, r = Rotation.from_matrix(R_world_cam_xyz).as_euler("ZYX", degrees=True)
        roll, pitch, yaw = r, -p, 90 - y
        roll_pitch_yaw = np.array([-roll, -pitch, yaw], np.float32)  # for some reason

        return {
            "t_c2w": t_world_cam.astype(np.float32),
            "roll_pitch_yaw": roll_pitch_yaw,
            "index": int(index_stem),
        }

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.image_paths[stage],
            self.data[stage],
            {self.dummy_scene_name: self.root},
            {self.dummy_scene_name: self.tile_manager},
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
        
        batch_size = cfg["batch_size"]
        if self.cfg.group_pano_views:
            if batch_size % 3 != 0:
                raise ValueError(f"Batch size ({batch_size}) must be a multiple of 3 for panorama training.")

        loader = torchdata.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            # shuffle=shuffle or (stage == "train"),
            shuffle = False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
            drop_last=True,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)

    def sequence_dataset(self, stage: str, **kwargs):
        keys = self.image_paths[stage]
        # group images by sequence (date/drive)
        seq2indices = defaultdict(list)
        for index, (_, date_drive, _) in enumerate(keys):
            seq2indices[date_drive].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(
                self.data[stage], indices, names=self.image_paths[stage], **kwargs
            )
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
        seq_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(seq_keys))
            seq_keys = [seq_keys[i] for i in perm]
        key_indices = [i for key in seq_keys for i in chunk2idx[key]]
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
        return loader, seq_keys, chunk2idx

class KittiAugmentDataModule(pl.LightningDataModule):
    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "kitti",
        # paths and fetch
        "data_dir": DATASETS_PATH / "kitti_extracted",
        "original_data_dir": DATASETS_PATH / "kitti",
        "tiles_filename": "tiles.pkl",
        "splits": {
            "train": "train_files_augment.txt",
            "val": "test1_files_augment.txt",
            "test": "test2_files_augment.txt",
        },
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 2, "num_workers": 0},
        },
        "max_num_val": 500,
        "selection_subset_val": "furthest",
        "drop_train_too_close_to_val": 5.0,
        "skip_frames": 1,
        "camera_index": 2,
        # overwrite
        "crop_size_meters": 64,
        "max_init_error": 20,
        #"max_init_error_rotation": 10,
        "max_init_error_rotation_eval": 10,
        "add_map_mask": True,
        "mask_pad": 2,
        "target_focal_length": 256,
        "group_pano_views": True,
    }
    dummy_scene_name = "kitti_extracted"

    def __init__(self, cfg, tile_manager: Optional[TileManager] = None):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.original_root = Path(self.cfg.original_data_dir)
        self.tile_manager = tile_manager
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")
        assert self.cfg.selection_subset_val in ["random", "furthest"]
        self.splits = {}
        self.shifts = {}
        self.calibrations = {}
        self.data = {}
        self.image_paths = {}
        self._is_setup = False
        self.num_views = 2
    '''
    def prepare_data(self):
        if not (self.root.exists() and (self.root / ".downloaded").exists()):
            raise FileNotFoundError(
                "Cannot find the KITTI dataset, run maploc.data.kitti.prepare"
            )
    '''
    def _group_views_by_panorama(self, names):
        pano_groups = defaultdict(list)
        for name in names:
            # name is a tuple: (date, drive, index_file)
            # e.g. ('2011_09_26', '2011_09_26_drive_0005_sync', '0000000048_left.png')
            index_file = name[-1]
            pano_id = f"{name[0]}_{name[1]}_{Path(index_file).stem.rsplit('_', 1)[0]}"
            pano_groups[pano_id].append(name)

        grouped_names = []
        for pano_id in sorted(pano_groups.keys()):
            views = sorted(pano_groups[pano_id], key=lambda n: n[-1])
            if len(views) != self.num_views:
                logger.warning(f"Panorama {pano_id} has {len(views)} views, expected {self.num_views}.")
                # continue
            grouped_names.extend(views)
        return grouped_names

    def parse_split(self, split_arg):
        if isinstance(split_arg, str):
            names, shifts = parse_split_file(self.root / split_arg)
        elif isinstance(split_arg, collections.abc.Sequence):
            names = []
            shifts = None
            for date_drive in split_arg:
                data_dir = (
                    self.root / date_drive / f"image_{self.cfg.camera_index:02}/data"
                )
                assert data_dir.exists(), data_dir
                date_drive = tuple(date_drive.split("/"))
                n = sorted(date_drive + (p.name,) for p in data_dir.glob("*.png"))
                names.extend(n[:: self.cfg.skip_frames])
        else:
            raise ValueError(split_arg)
        return names, shifts

    def setup(self, stage: Optional[str] = None):
        if self._is_setup:
            return
        
        if stage == "fit":
            stages = ["train", "val"]
        elif stage is None:
            stages = ["train", "val", "test"]
        else:
            stages = [stage]
        for stage_ in stages:
            self.splits[stage_], self.shifts[stage_] = self.parse_split(
                self.cfg.splits[stage_]
            )

        if self.cfg.group_pano_views:
            for stage_ in stages:
                self.splits[stage_] = self._group_views_by_panorama(self.splits[stage_])

        do_val_subset = "val" in stages and self.cfg.max_num_val is not None
        if do_val_subset and self.cfg.selection_subset_val == "random":
            select = np.random.RandomState(self.cfg.seed).choice(
                len(self.splits["val"]), self.cfg.max_num_val, replace=False
            )
            self.splits["val"] = [self.splits["val"][i] for i in select]
            if self.shifts["val"] is not None:
                self.shifts["val"] = self.shifts["val"][select]
        dates = {d for ns in self.splits.values() for d, _, _ in ns}
        for d in dates:
            self.calibrations[d] = get_camera_calibration(
                self.original_root / d, self.cfg.camera_index
            )
        if self.tile_manager is None:
            logger.info("Loading the tile manager...")
            self.tile_manager = TileManager.load(self.root / self.cfg.tiles_filename)
        self.cfg.num_classes = {k: len(g) for k, g in self.tile_manager.groups.items()}
        self.cfg.pixel_per_meter = self.tile_manager.ppm

        # pack all attributes in a single tensor to optimize memory access
        self.pack_data(stages)

        dists = None
        if do_val_subset and self.cfg.selection_subset_val == "furthest":
            num_panos_val = len(self.splits["val"]) // self.num_views
            num_panos_train = len(self.splits["train"]) // self.num_views
            # Get the position of the full view of each panorama [start:stop:step]
            pos_val = self.data["val"]["t_c2w"][ : self.num_views * num_panos_val : self.num_views, :2].double()
            pos_train = self.data["train"]["t_c2w"][ : self.num_views * num_panos_train : self.num_views, :2].double()

            dists = torch.cdist(pos_val, pos_train)

            min_dists = dists.min(1).values
            max_num_val_panos = self.cfg.max_num_val // self.num_views
            select_panos = torch.argsort(min_dists)[-max_num_val_panos:]
            
            # Convert pano indices to item indices (num_views per pano)
            select = torch.cat([select_panos * self.num_views + i for i in range(self.num_views)]).sort().values

            dists = dists[select_panos]
            self.splits["val"] = [self.splits["val"][i] for i in select]
            if self.shifts["val"] is not None:
                self.shifts["val"] = self.shifts["val"][select]
            for k in list(self.data["val"]):
                if k != "cameras":
                    self.data["val"][k] = self.data["val"][k][select]
            self.image_paths["val"] = self.image_paths["val"][select]

        if "train" in stages and self.cfg.drop_train_too_close_to_val is not None:
            if dists is None:
                num_panos_val = len(self.splits["val"]) // self.num_views
                num_panos_train = len(self.splits["train"]) // self.num_views
                pos_val = self.data["val"]["t_c2w"][: self.num_views * num_panos_val : self.num_views, :2].double()
                pos_train = self.data["train"]["t_c2w"][: self.num_views * num_panos_train : self.num_views, :2].double()
                dists = torch.cdist(pos_val, pos_train)

            drop_panos = torch.any(dists < self.cfg.drop_train_too_close_to_val, 0)
            select_panos = torch.where(~drop_panos)[0]
            
            select = torch.cat([select_panos * self.num_views + i for i in range(self.num_views)]).sort().values

            logger.info(
                "Dropping %d (%f %%) images that are too close to validation images.",
                drop_panos.sum() * self.num_views,
                drop_panos.float().mean(),
            )
            self.splits["train"] = [self.splits["train"][i] for i in select]
            if self.shifts["train"] is not None:
                self.shifts["train"] = self.shifts["train"][select]
            for k in list(self.data["train"]):
                if k != "cameras":
                    self.data["train"][k] = self.data["train"][k][select]
            self.image_paths["train"] = self.image_paths["train"][select]
        
        self._is_setup = True

    def pack_data(self, stages):
        for stage in stages:
            names = []
            data = {}
            for i, (date, drive, index) in enumerate(self.splits[stage]):
                d = self.get_frame_data(date, drive, index)
                for k, v in d.items():
                    if i == 0:
                        data[k] = []
                    data[k].append(v)
                path = f"{date}/{drive}/image_{self.cfg.camera_index:02}/data/{index}"
                names.append((self.dummy_scene_name, f"{date}/{drive}", path))
            for k in list(data):
                data[k] = torch.from_numpy(np.stack(data[k]))
            data["camera_id"] = np.full(len(names), self.cfg.camera_index)

            sequences = {date_drive for _, date_drive, _ in names}
            data["cameras"] = {
                self.dummy_scene_name: {
                    seq: {
                        self.cfg.camera_index: self.calibrations[seq.split("/")[0]][0]
                    }
                    for seq in sequences
                }
            }
            shifts = self.shifts[stage]
            if shifts is not None:
                data["shifts"] = torch.from_numpy(shifts.astype(np.float32))
            self.data[stage] = data
            self.image_paths[stage] = np.array(names)

    def get_frame_data(self, date, drive, index):
        _, R_cam_gps, t_cam_gps = self.calibrations[date]

        # Transform the GPS pose to the camera pose
        index_stem = Path(index).stem.split("_")[0]
        gps_path = (
            self.original_root / date / drive / "oxts/data" / Path(index_stem).with_suffix(".txt")
        )
        _, R_world_gps, t_world_gps = parse_gps_file(
            gps_path, self.tile_manager.projection
        )
        R_world_cam = R_world_gps @ R_cam_gps.T
        t_world_cam = t_world_gps - R_world_gps @ R_cam_gps.T @ t_cam_gps
        # Some voodoo to extract correct Euler angles from R_world_cam
        R_cv_xyz = Rotation.from_euler("YX", [-90, 90], degrees=True).as_matrix()
        R_world_cam_xyz = R_world_cam @ R_cv_xyz
        y, p, r = Rotation.from_matrix(R_world_cam_xyz).as_euler("ZYX", degrees=True)
        roll, pitch, yaw = r, -p, 90 - y
        roll_pitch_yaw = np.array([-roll, -pitch, yaw], np.float32)  # for some reason

        return {
            "t_c2w": t_world_cam.astype(np.float32),
            "roll_pitch_yaw": roll_pitch_yaw,
            "index": int(index_stem),
        }

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.image_paths[stage],
            self.data[stage],
            {self.dummy_scene_name: self.root},
            {self.dummy_scene_name: self.tile_manager},
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
        
        batch_size = cfg["batch_size"]
        if self.cfg.group_pano_views:
            if batch_size % self.num_views != 0:
                raise ValueError(f"Batch size ({batch_size}) must be a multiple of {self.num_views} for panorama training.")

        loader = torchdata.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            # shuffle=shuffle or (stage == "train"),
            shuffle = False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
            drop_last=True,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)

    def sequence_dataset(self, stage: str, **kwargs):
        keys = self.image_paths[stage]
        # group images by sequence (date/drive)
        seq2indices = defaultdict(list)
        for index, (_, date_drive, _) in enumerate(keys):
            seq2indices[date_drive].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(
                self.data[stage], indices, names=self.image_paths[stage], **kwargs
            )
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
        seq_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(seq_keys))
            seq_keys = [seq_keys[i] for i in perm]
        key_indices = [i for key in seq_keys for i in chunk2idx[key]]
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
        return loader, seq_keys, chunk2idx

class KittiDataModule(pl.LightningDataModule):
    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "kitti",
        # paths and fetch
        "data_dir": DATASETS_PATH / "kitti",
        "tiles_filename": "tiles.pkl",
        "splits": {
            "train": "origin_train_files.txt",
            "val": "origin_test1_files.txt",
            "test": "origin_test2_files.txt",
        },
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 1, "num_workers": 0},
        },
        "max_num_val": 500,
        "selection_subset_val": "furthest",
        "drop_train_too_close_to_val": 5.0,
        "skip_frames": 1,
        "camera_index": 2,
        # overwrite
        "crop_size_meters": 64,
        "max_init_error": 20,
        #"max_init_error_rotation": 10,
        "max_init_error_rotation_eval": 10,
        "add_map_mask": True,
        "mask_pad": 2,
        "target_focal_length": 256,
    }
    dummy_scene_name = "kitti"

    def __init__(self, cfg, tile_manager: Optional[TileManager] = None):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.tile_manager = tile_manager
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")
        assert self.cfg.selection_subset_val in ["random", "furthest"]
        self.splits = {}
        self.shifts = {}
        self.calibrations = {}
        self.data = {}
        self.image_paths = {}
    '''
    def prepare_data(self):
        if not (self.root.exists() and (self.root / ".downloaded").exists()):
            raise FileNotFoundError(
                "Cannot find the KITTI dataset, run maploc.data.kitti.prepare"
            )
    '''
    def parse_split(self, split_arg):
        if isinstance(split_arg, str):
            names, shifts = parse_split_file(self.root / split_arg)
        elif isinstance(split_arg, collections.abc.Sequence):
            names = []
            shifts = None
            for date_drive in split_arg:
                data_dir = (
                    self.root / date_drive / f"image_{self.cfg.camera_index:02}/data"
                )
                assert data_dir.exists(), data_dir
                date_drive = tuple(date_drive.split("/"))
                n = sorted(date_drive + (p.name,) for p in data_dir.glob("*.png"))
                names.extend(n[:: self.cfg.skip_frames])
        else:
            raise ValueError(split_arg)
        return names, shifts

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            stages = ["train", "val"]
        elif stage is None:
            stages = ["train", "val", "test"]
        else:
            stages = [stage]
        for stage in stages:
            self.splits[stage], self.shifts[stage] = self.parse_split(
                self.cfg.splits[stage]
            )
        do_val_subset = "val" in stages and self.cfg.max_num_val is not None
        if do_val_subset and self.cfg.selection_subset_val == "random":
            select = np.random.RandomState(self.cfg.seed).choice(
                len(self.splits["val"]), self.cfg.max_num_val, replace=False
            )
            self.splits["val"] = [self.splits["val"][i] for i in select]
            if self.shifts["val"] is not None:
                self.shifts["val"] = self.shifts["val"][select]
        dates = {d for ns in self.splits.values() for d, _, _ in ns}
        for d in dates:
            self.calibrations[d] = get_camera_calibration(
                self.root / d, self.cfg.camera_index
            )
        if self.tile_manager is None:
            logger.info("Loading the tile manager...")
            self.tile_manager = TileManager.load(self.root / self.cfg.tiles_filename)
        self.cfg.num_classes = {k: len(g) for k, g in self.tile_manager.groups.items()}
        self.cfg.pixel_per_meter = self.tile_manager.ppm

        # pack all attributes in a single tensor to optimize memory access
        self.pack_data(stages)

        dists = None
        if do_val_subset and self.cfg.selection_subset_val == "furthest":
            dists = torch.cdist(
                self.data["val"]["t_c2w"][:, :2].double(),
                self.data["train"]["t_c2w"][:, :2].double(),
            )
            min_dists = dists.min(1).values
            select = torch.argsort(min_dists)[-self.cfg.max_num_val :]
            dists = dists[select]
            self.splits["val"] = [self.splits["val"][i] for i in select]
            if self.shifts["val"] is not None:
                self.shifts["val"] = self.shifts["val"][select]
            for k in list(self.data["val"]):
                if k != "cameras":
                    self.data["val"][k] = self.data["val"][k][select]
            self.image_paths["val"] = self.image_paths["val"][select]

        if "train" in stages and self.cfg.drop_train_too_close_to_val is not None:
            if dists is None:
                dists = torch.cdist(
                    self.data["val"]["t_c2w"][:, :2].double(),
                    self.data["train"]["t_c2w"][:, :2].double(),
                )
            drop = torch.any(dists < self.cfg.drop_train_too_close_to_val, 0)
            select = torch.where(~drop)[0]
            logger.info(
                "Dropping %d (%f %%) images that are too close to validation images.",
                drop.sum(),
                drop.float().mean(),
            )
            self.splits["train"] = [self.splits["train"][i] for i in select]
            if self.shifts["train"] is not None:
                self.shifts["train"] = self.shifts["train"][select]
            for k in list(self.data["train"]):
                if k != "cameras":
                    self.data["train"][k] = self.data["train"][k][select]
            self.image_paths["train"] = self.image_paths["train"][select]

    def pack_data(self, stages):
        for stage in stages:
            names = []
            data = {}
            for i, (date, drive, index) in enumerate(self.splits[stage]):
                d = self.get_frame_data(date, drive, index)
                for k, v in d.items():
                    if i == 0:
                        data[k] = []
                    data[k].append(v)
                path = f"{date}/{drive}/image_{self.cfg.camera_index:02}/data/{index}"
                names.append((self.dummy_scene_name, f"{date}/{drive}", path))
            for k in list(data):
                data[k] = torch.from_numpy(np.stack(data[k]))
            data["camera_id"] = np.full(len(names), self.cfg.camera_index)

            sequences = {date_drive for _, date_drive, _ in names}
            data["cameras"] = {
                self.dummy_scene_name: {
                    seq: {
                        self.cfg.camera_index: self.calibrations[seq.split("/")[0]][0]
                    }
                    for seq in sequences
                }
            }
            shifts = self.shifts[stage]
            if shifts is not None:
                data["shifts"] = torch.from_numpy(shifts.astype(np.float32))
            self.data[stage] = data
            self.image_paths[stage] = np.array(names)

    def get_frame_data(self, date, drive, index):
        _, R_cam_gps, t_cam_gps = self.calibrations[date]

        # Transform the GPS pose to the camera pose
        gps_path = (
            self.root / date / drive / "oxts/data" / Path(index).with_suffix(".txt")
        )
        _, R_world_gps, t_world_gps = parse_gps_file(
            gps_path, self.tile_manager.projection
        )
        R_world_cam = R_world_gps @ R_cam_gps.T
        t_world_cam = t_world_gps - R_world_gps @ R_cam_gps.T @ t_cam_gps
        # Some voodoo to extract correct Euler angles from R_world_cam
        R_cv_xyz = Rotation.from_euler("YX", [-90, 90], degrees=True).as_matrix()
        R_world_cam_xyz = R_world_cam @ R_cv_xyz
        y, p, r = Rotation.from_matrix(R_world_cam_xyz).as_euler("ZYX", degrees=True)
        roll, pitch, yaw = r, -p, 90 - y
        roll_pitch_yaw = np.array([-roll, -pitch, yaw], np.float32)  # for some reason

        return {
            "t_c2w": t_world_cam.astype(np.float32),
            "roll_pitch_yaw": roll_pitch_yaw,
            "index": int(index.split(".")[0]),
        }

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.image_paths[stage],
            self.data[stage],
            {self.dummy_scene_name: self.root},
            {self.dummy_scene_name: self.tile_manager},
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
        keys = self.image_paths[stage]
        # group images by sequence (date/drive)
        seq2indices = defaultdict(list)
        for index, (_, date_drive, _) in enumerate(keys):
            seq2indices[date_drive].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(
                self.data[stage], indices, names=self.image_paths[stage], **kwargs
            )
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
        seq_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(seq_keys))
            seq_keys = [seq_keys[i] for i in perm]
        key_indices = [i for key in seq_keys for i in chunk2idx[key]]
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
        return loader, seq_keys, chunk2idx

class KittiRainyDataModule(pl.LightningDataModule):
    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "kitti_rainy",
        # paths and fetch
        "data_dir": DATASETS_PATH / "weather_noise/rainy",
        "tiles_filename": "tiles.pkl",
        "splits": {
            "train": "train_files.txt",
            "val": "test1_files.txt",
            "test": "test2_files.txt",
        },
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 1, "num_workers": 0},
        },
        "max_num_val": 500,
        "selection_subset_val": "furthest",
        "drop_train_too_close_to_val": 5.0,
        "skip_frames": 1,
        "camera_index": 2,
        # overwrite
        "crop_size_meters": 64,
        "max_init_error": 20,
        #"max_init_error_rotation": 10,
        "max_init_error_rotation_eval": 10,
        "add_map_mask": True,
        "mask_pad": 2,
        "target_focal_length": 256,
    }
    dummy_scene_name = "kitti_rainy"

    def __init__(self, cfg, tile_manager: Optional[TileManager] = None):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.tile_manager = tile_manager
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")
        assert self.cfg.selection_subset_val in ["random", "furthest"]
        self.splits = {}
        self.shifts = {}
        self.calibrations = {}
        self.data = {}
        self.image_paths = {}

    def prepare_data(self):
        if not (self.root.exists() and (self.root / ".downloaded").exists()):
            raise FileNotFoundError(
                "Cannot find the KITTI dataset, run maploc.data.kitti.prepare"
            )

    def parse_split(self, split_arg):
        if isinstance(split_arg, str):
            names, shifts = parse_split_file(self.root / split_arg)
        elif isinstance(split_arg, collections.abc.Sequence):
            names = []
            shifts = None
            for date_drive in split_arg:
                data_dir = (
                    self.root / date_drive / f"image_{self.cfg.camera_index:02}/data"
                )
                assert data_dir.exists(), data_dir
                date_drive = tuple(date_drive.split("/"))
                n = sorted(date_drive + (p.name,) for p in data_dir.glob("*.png"))
                names.extend(n[:: self.cfg.skip_frames])
        else:
            raise ValueError(split_arg)
        return names, shifts

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            stages = ["train", "val"]
        elif stage is None:
            stages = ["train", "val", "test"]
        else:
            stages = [stage]
        for stage in stages:
            self.splits[stage], self.shifts[stage] = self.parse_split(
                self.cfg.splits[stage]
            )
        do_val_subset = "val" in stages and self.cfg.max_num_val is not None
        if do_val_subset and self.cfg.selection_subset_val == "random":
            select = np.random.RandomState(self.cfg.seed).choice(
                len(self.splits["val"]), self.cfg.max_num_val, replace=False
            )
            self.splits["val"] = [self.splits["val"][i] for i in select]
            if self.shifts["val"] is not None:
                self.shifts["val"] = self.shifts["val"][select]
        dates = {d for ns in self.splits.values() for d, _, _ in ns}
        for d in dates:
            self.calibrations[d] = get_camera_calibration(
                self.root / d, self.cfg.camera_index
            )
        if self.tile_manager is None:
            logger.info("Loading the tile manager...")
            self.tile_manager = TileManager.load(self.root / self.cfg.tiles_filename)
        self.cfg.num_classes = {k: len(g) for k, g in self.tile_manager.groups.items()}
        self.cfg.pixel_per_meter = self.tile_manager.ppm

        # pack all attributes in a single tensor to optimize memory access
        self.pack_data(stages)

        dists = None
        if do_val_subset and self.cfg.selection_subset_val == "furthest":
            dists = torch.cdist(
                self.data["val"]["t_c2w"][:, :2].double(),
                self.data["train"]["t_c2w"][:, :2].double(),
            )
            min_dists = dists.min(1).values
            select = torch.argsort(min_dists)[-self.cfg.max_num_val :]
            dists = dists[select]
            self.splits["val"] = [self.splits["val"][i] for i in select]
            if self.shifts["val"] is not None:
                self.shifts["val"] = self.shifts["val"][select]
            for k in list(self.data["val"]):
                if k != "cameras":
                    self.data["val"][k] = self.data["val"][k][select]
            self.image_paths["val"] = self.image_paths["val"][select]

        if "train" in stages and self.cfg.drop_train_too_close_to_val is not None:
            if dists is None:
                dists = torch.cdist(
                    self.data["val"]["t_c2w"][:, :2].double(),
                    self.data["train"]["t_c2w"][:, :2].double(),
                )
            drop = torch.any(dists < self.cfg.drop_train_too_close_to_val, 0)
            select = torch.where(~drop)[0]
            logger.info(
                "Dropping %d (%f %%) images that are too close to validation images.",
                drop.sum(),
                drop.float().mean(),
            )
            self.splits["train"] = [self.splits["train"][i] for i in select]
            if self.shifts["train"] is not None:
                self.shifts["train"] = self.shifts["train"][select]
            for k in list(self.data["train"]):
                if k != "cameras":
                    self.data["train"][k] = self.data["train"][k][select]
            self.image_paths["train"] = self.image_paths["train"][select]

    def pack_data(self, stages):
        for stage in stages:
            names = []
            data = {}
            for i, (date, drive, index) in enumerate(self.splits[stage]):
                d = self.get_frame_data(date, drive, index)
                for k, v in d.items():
                    if i == 0:
                        data[k] = []
                    data[k].append(v)
                path = f"{date}/{drive}/image_{self.cfg.camera_index:02}/data/{index}"
                names.append((self.dummy_scene_name, f"{date}/{drive}", path))
            for k in list(data):
                data[k] = torch.from_numpy(np.stack(data[k]))
            data["camera_id"] = np.full(len(names), self.cfg.camera_index)

            sequences = {date_drive for _, date_drive, _ in names}
            data["cameras"] = {
                self.dummy_scene_name: {
                    seq: {
                        self.cfg.camera_index: self.calibrations[seq.split("/")[0]][0]
                    }
                    for seq in sequences
                }
            }
            shifts = self.shifts[stage]
            if shifts is not None:
                data["shifts"] = torch.from_numpy(shifts.astype(np.float32))
            self.data[stage] = data
            self.image_paths[stage] = np.array(names)

    def get_frame_data(self, date, drive, index):
        _, R_cam_gps, t_cam_gps = self.calibrations[date]

        # Transform the GPS pose to the camera pose
        gps_path = (
            self.root / date / drive / "oxts/data" / Path(index).with_suffix(".txt")
        )
        _, R_world_gps, t_world_gps = parse_gps_file(
            gps_path, self.tile_manager.projection
        )
        R_world_cam = R_world_gps @ R_cam_gps.T
        t_world_cam = t_world_gps - R_world_gps @ R_cam_gps.T @ t_cam_gps
        # Some voodoo to extract correct Euler angles from R_world_cam
        R_cv_xyz = Rotation.from_euler("YX", [-90, 90], degrees=True).as_matrix()
        R_world_cam_xyz = R_world_cam @ R_cv_xyz
        y, p, r = Rotation.from_matrix(R_world_cam_xyz).as_euler("ZYX", degrees=True)
        roll, pitch, yaw = r, -p, 90 - y
        roll_pitch_yaw = np.array([-roll, -pitch, yaw], np.float32)  # for some reason

        return {
            "t_c2w": t_world_cam.astype(np.float32),
            "roll_pitch_yaw": roll_pitch_yaw,
            "index": int(index.split(".")[0]),
        }

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.image_paths[stage],
            self.data[stage],
            {self.dummy_scene_name: self.root},
            {self.dummy_scene_name: self.tile_manager},
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
        keys = self.image_paths[stage]
        # group images by sequence (date/drive)
        seq2indices = defaultdict(list)
        for index, (_, date_drive, _) in enumerate(keys):
            seq2indices[date_drive].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(
                self.data[stage], indices, names=self.image_paths[stage], **kwargs
            )
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
        seq_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(seq_keys))
            seq_keys = [seq_keys[i] for i in perm]
        key_indices = [i for key in seq_keys for i in chunk2idx[key]]
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
        return loader, seq_keys, chunk2idx

class KittiNightDataModule(pl.LightningDataModule):
    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "kitti_night",
        # paths and fetch
        "data_dir": DATASETS_PATH / "weather_noise/night",
        "tiles_filename": "tiles.pkl",
        "splits": {
            "train": "train_files.txt",
            "val": "test1_files.txt",
            "test": "test2_files.txt",
        },
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 1, "num_workers": 0},
        },
        "max_num_val": 500,
        "selection_subset_val": "furthest",
        "drop_train_too_close_to_val": 5.0,
        "skip_frames": 1,
        "camera_index": 2,
        # overwrite
        "crop_size_meters": 64,
        "max_init_error": 20,
        #"max_init_error_rotation": 10,
        "max_init_error_rotation_eval": 10,
        "add_map_mask": True,
        "mask_pad": 2,
        "target_focal_length": 256,
    }
    dummy_scene_name = "kitti_night"

    def __init__(self, cfg, tile_manager: Optional[TileManager] = None):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.tile_manager = tile_manager
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")
        assert self.cfg.selection_subset_val in ["random", "furthest"]
        self.splits = {}
        self.shifts = {}
        self.calibrations = {}
        self.data = {}
        self.image_paths = {}

    def prepare_data(self):
        if not (self.root.exists() and (self.root / ".downloaded").exists()):
            raise FileNotFoundError(
                "Cannot find the KITTI dataset, run maploc.data.kitti.prepare"
            )

    def parse_split(self, split_arg):
        if isinstance(split_arg, str):
            names, shifts = parse_split_file(self.root / split_arg)
        elif isinstance(split_arg, collections.abc.Sequence):
            names = []
            shifts = None
            for date_drive in split_arg:
                data_dir = (
                    self.root / date_drive / f"image_{self.cfg.camera_index:02}/data"
                )
                assert data_dir.exists(), data_dir
                date_drive = tuple(date_drive.split("/"))
                n = sorted(date_drive + (p.name,) for p in data_dir.glob("*.png"))
                names.extend(n[:: self.cfg.skip_frames])
        else:
            raise ValueError(split_arg)
        return names, shifts

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            stages = ["train", "val"]
        elif stage is None:
            stages = ["train", "val", "test"]
        else:
            stages = [stage]
        for stage in stages:
            self.splits[stage], self.shifts[stage] = self.parse_split(
                self.cfg.splits[stage]
            )
        do_val_subset = "val" in stages and self.cfg.max_num_val is not None
        if do_val_subset and self.cfg.selection_subset_val == "random":
            select = np.random.RandomState(self.cfg.seed).choice(
                len(self.splits["val"]), self.cfg.max_num_val, replace=False
            )
            self.splits["val"] = [self.splits["val"][i] for i in select]
            if self.shifts["val"] is not None:
                self.shifts["val"] = self.shifts["val"][select]
        dates = {d for ns in self.splits.values() for d, _, _ in ns}
        for d in dates:
            self.calibrations[d] = get_camera_calibration(
                self.root / d, self.cfg.camera_index
            )
        if self.tile_manager is None:
            logger.info("Loading the tile manager...")
            self.tile_manager = TileManager.load(self.root / self.cfg.tiles_filename)
        self.cfg.num_classes = {k: len(g) for k, g in self.tile_manager.groups.items()}
        self.cfg.pixel_per_meter = self.tile_manager.ppm

        # pack all attributes in a single tensor to optimize memory access
        self.pack_data(stages)

        dists = None
        if do_val_subset and self.cfg.selection_subset_val == "furthest":
            dists = torch.cdist(
                self.data["val"]["t_c2w"][:, :2].double(),
                self.data["train"]["t_c2w"][:, :2].double(),
            )
            min_dists = dists.min(1).values
            select = torch.argsort(min_dists)[-self.cfg.max_num_val :]
            dists = dists[select]
            self.splits["val"] = [self.splits["val"][i] for i in select]
            if self.shifts["val"] is not None:
                self.shifts["val"] = self.shifts["val"][select]
            for k in list(self.data["val"]):
                if k != "cameras":
                    self.data["val"][k] = self.data["val"][k][select]
            self.image_paths["val"] = self.image_paths["val"][select]

        if "train" in stages and self.cfg.drop_train_too_close_to_val is not None:
            if dists is None:
                dists = torch.cdist(
                    self.data["val"]["t_c2w"][:, :2].double(),
                    self.data["train"]["t_c2w"][:, :2].double(),
                )
            drop = torch.any(dists < self.cfg.drop_train_too_close_to_val, 0)
            select = torch.where(~drop)[0]
            logger.info(
                "Dropping %d (%f %%) images that are too close to validation images.",
                drop.sum(),
                drop.float().mean(),
            )
            self.splits["train"] = [self.splits["train"][i] for i in select]
            if self.shifts["train"] is not None:
                self.shifts["train"] = self.shifts["train"][select]
            for k in list(self.data["train"]):
                if k != "cameras":
                    self.data["train"][k] = self.data["train"][k][select]
            self.image_paths["train"] = self.image_paths["train"][select]

    def pack_data(self, stages):
        for stage in stages:
            names = []
            data = {}
            for i, (date, drive, index) in enumerate(self.splits[stage]):
                d = self.get_frame_data(date, drive, index)
                for k, v in d.items():
                    if i == 0:
                        data[k] = []
                    data[k].append(v)
                path = f"{date}/{drive}/image_{self.cfg.camera_index:02}/data/{index}"
                names.append((self.dummy_scene_name, f"{date}/{drive}", path))
            for k in list(data):
                data[k] = torch.from_numpy(np.stack(data[k]))
            data["camera_id"] = np.full(len(names), self.cfg.camera_index)

            sequences = {date_drive for _, date_drive, _ in names}
            data["cameras"] = {
                self.dummy_scene_name: {
                    seq: {
                        self.cfg.camera_index: self.calibrations[seq.split("/")[0]][0]
                    }
                    for seq in sequences
                }
            }
            shifts = self.shifts[stage]
            if shifts is not None:
                data["shifts"] = torch.from_numpy(shifts.astype(np.float32))
            self.data[stage] = data
            self.image_paths[stage] = np.array(names)

    def get_frame_data(self, date, drive, index):
        _, R_cam_gps, t_cam_gps = self.calibrations[date]

        # Transform the GPS pose to the camera pose
        gps_path = (
            self.root / date / drive / "oxts/data" / Path(index).with_suffix(".txt")
        )
        _, R_world_gps, t_world_gps = parse_gps_file(
            gps_path, self.tile_manager.projection
        )
        R_world_cam = R_world_gps @ R_cam_gps.T
        t_world_cam = t_world_gps - R_world_gps @ R_cam_gps.T @ t_cam_gps
        # Some voodoo to extract correct Euler angles from R_world_cam
        R_cv_xyz = Rotation.from_euler("YX", [-90, 90], degrees=True).as_matrix()
        R_world_cam_xyz = R_world_cam @ R_cv_xyz
        y, p, r = Rotation.from_matrix(R_world_cam_xyz).as_euler("ZYX", degrees=True)
        roll, pitch, yaw = r, -p, 90 - y
        roll_pitch_yaw = np.array([-roll, -pitch, yaw], np.float32)  # for some reason

        return {
            "t_c2w": t_world_cam.astype(np.float32),
            "roll_pitch_yaw": roll_pitch_yaw,
            "index": int(index.split(".")[0]),
        }

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.image_paths[stage],
            self.data[stage],
            {self.dummy_scene_name: self.root},
            {self.dummy_scene_name: self.tile_manager},
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
        keys = self.image_paths[stage]
        # group images by sequence (date/drive)
        seq2indices = defaultdict(list)
        for index, (_, date_drive, _) in enumerate(keys):
            seq2indices[date_drive].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(
                self.data[stage], indices, names=self.image_paths[stage], **kwargs
            )
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
        seq_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(seq_keys))
            seq_keys = [seq_keys[i] for i in perm]
        key_indices = [i for key in seq_keys for i in chunk2idx[key]]
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
        return loader, seq_keys, chunk2idx

class KittiFoggyDataModule(pl.LightningDataModule):
    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "kitti_foggy",
        # paths and fetch
        "data_dir": DATASETS_PATH / "weather_noise/foggy",
        "tiles_filename": "tiles.pkl",
        "splits": {
            "train": "train_files.txt",
            "val": "test1_files.txt",
            "test": "test2_files.txt",
        },
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 1, "num_workers": 0},
        },
        "max_num_val": 500,
        "selection_subset_val": "furthest",
        "drop_train_too_close_to_val": 5.0,
        "skip_frames": 1,
        "camera_index": 2,
        # overwrite
        "crop_size_meters": 64,
        "max_init_error": 20,
        #"max_init_error_rotation": 10,
        "max_init_error_rotation_eval": 10,
        "add_map_mask": True,
        "mask_pad": 2,
        "target_focal_length": 256,
    }
    dummy_scene_name = "kitti_foggy"

    def __init__(self, cfg, tile_manager: Optional[TileManager] = None):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.tile_manager = tile_manager
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")
        assert self.cfg.selection_subset_val in ["random", "furthest"]
        self.splits = {}
        self.shifts = {}
        self.calibrations = {}
        self.data = {}
        self.image_paths = {}

    def prepare_data(self):
        if not (self.root.exists() and (self.root / ".downloaded").exists()):
            raise FileNotFoundError(
                "Cannot find the KITTI dataset, run maploc.data.kitti.prepare"
            )

    def parse_split(self, split_arg):
        if isinstance(split_arg, str):
            names, shifts = parse_split_file(self.root / split_arg)
        elif isinstance(split_arg, collections.abc.Sequence):
            names = []
            shifts = None
            for date_drive in split_arg:
                data_dir = (
                    self.root / date_drive / f"image_{self.cfg.camera_index:02}/data"
                )
                assert data_dir.exists(), data_dir
                date_drive = tuple(date_drive.split("/"))
                n = sorted(date_drive + (p.name,) for p in data_dir.glob("*.png"))
                names.extend(n[:: self.cfg.skip_frames])
        else:
            raise ValueError(split_arg)
        return names, shifts

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            stages = ["train", "val"]
        elif stage is None:
            stages = ["train", "val", "test"]
        else:
            stages = [stage]
        for stage in stages:
            self.splits[stage], self.shifts[stage] = self.parse_split(
                self.cfg.splits[stage]
            )
        do_val_subset = "val" in stages and self.cfg.max_num_val is not None
        if do_val_subset and self.cfg.selection_subset_val == "random":
            select = np.random.RandomState(self.cfg.seed).choice(
                len(self.splits["val"]), self.cfg.max_num_val, replace=False
            )
            self.splits["val"] = [self.splits["val"][i] for i in select]
            if self.shifts["val"] is not None:
                self.shifts["val"] = self.shifts["val"][select]
        dates = {d for ns in self.splits.values() for d, _, _ in ns}
        for d in dates:
            self.calibrations[d] = get_camera_calibration(
                self.root / d, self.cfg.camera_index
            )
        if self.tile_manager is None:
            logger.info("Loading the tile manager...")
            self.tile_manager = TileManager.load(self.root / self.cfg.tiles_filename)
        self.cfg.num_classes = {k: len(g) for k, g in self.tile_manager.groups.items()}
        self.cfg.pixel_per_meter = self.tile_manager.ppm

        # pack all attributes in a single tensor to optimize memory access
        self.pack_data(stages)

        dists = None
        if do_val_subset and self.cfg.selection_subset_val == "furthest":
            dists = torch.cdist(
                self.data["val"]["t_c2w"][:, :2].double(),
                self.data["train"]["t_c2w"][:, :2].double(),
            )
            min_dists = dists.min(1).values
            select = torch.argsort(min_dists)[-self.cfg.max_num_val :]
            dists = dists[select]
            self.splits["val"] = [self.splits["val"][i] for i in select]
            if self.shifts["val"] is not None:
                self.shifts["val"] = self.shifts["val"][select]
            for k in list(self.data["val"]):
                if k != "cameras":
                    self.data["val"][k] = self.data["val"][k][select]
            self.image_paths["val"] = self.image_paths["val"][select]

        if "train" in stages and self.cfg.drop_train_too_close_to_val is not None:
            if dists is None:
                dists = torch.cdist(
                    self.data["val"]["t_c2w"][:, :2].double(),
                    self.data["train"]["t_c2w"][:, :2].double(),
                )
            drop = torch.any(dists < self.cfg.drop_train_too_close_to_val, 0)
            select = torch.where(~drop)[0]
            logger.info(
                "Dropping %d (%f %%) images that are too close to validation images.",
                drop.sum(),
                drop.float().mean(),
            )
            self.splits["train"] = [self.splits["train"][i] for i in select]
            if self.shifts["train"] is not None:
                self.shifts["train"] = self.shifts["train"][select]
            for k in list(self.data["train"]):
                if k != "cameras":
                    self.data["train"][k] = self.data["train"][k][select]
            self.image_paths["train"] = self.image_paths["train"][select]

    def pack_data(self, stages):
        for stage in stages:
            names = []
            data = {}
            for i, (date, drive, index) in enumerate(self.splits[stage]):
                d = self.get_frame_data(date, drive, index)
                for k, v in d.items():
                    if i == 0:
                        data[k] = []
                    data[k].append(v)
                path = f"{date}/{drive}/image_{self.cfg.camera_index:02}/data/{index}"
                names.append((self.dummy_scene_name, f"{date}/{drive}", path))
            for k in list(data):
                data[k] = torch.from_numpy(np.stack(data[k]))
            data["camera_id"] = np.full(len(names), self.cfg.camera_index)

            sequences = {date_drive for _, date_drive, _ in names}
            data["cameras"] = {
                self.dummy_scene_name: {
                    seq: {
                        self.cfg.camera_index: self.calibrations[seq.split("/")[0]][0]
                    }
                    for seq in sequences
                }
            }
            shifts = self.shifts[stage]
            if shifts is not None:
                data["shifts"] = torch.from_numpy(shifts.astype(np.float32))
            self.data[stage] = data
            self.image_paths[stage] = np.array(names)

    def get_frame_data(self, date, drive, index):
        _, R_cam_gps, t_cam_gps = self.calibrations[date]

        # Transform the GPS pose to the camera pose
        gps_path = (
            self.root / date / drive / "oxts/data" / Path(index).with_suffix(".txt")
        )
        _, R_world_gps, t_world_gps = parse_gps_file(
            gps_path, self.tile_manager.projection
        )
        R_world_cam = R_world_gps @ R_cam_gps.T
        t_world_cam = t_world_gps - R_world_gps @ R_cam_gps.T @ t_cam_gps
        # Some voodoo to extract correct Euler angles from R_world_cam
        R_cv_xyz = Rotation.from_euler("YX", [-90, 90], degrees=True).as_matrix()
        R_world_cam_xyz = R_world_cam @ R_cv_xyz
        y, p, r = Rotation.from_matrix(R_world_cam_xyz).as_euler("ZYX", degrees=True)
        roll, pitch, yaw = r, -p, 90 - y
        roll_pitch_yaw = np.array([-roll, -pitch, yaw], np.float32)  # for some reason

        return {
            "t_c2w": t_world_cam.astype(np.float32),
            "roll_pitch_yaw": roll_pitch_yaw,
            "index": int(index.split(".")[0]),
        }

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.image_paths[stage],
            self.data[stage],
            {self.dummy_scene_name: self.root},
            {self.dummy_scene_name: self.tile_manager},
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
        keys = self.image_paths[stage]
        # group images by sequence (date/drive)
        seq2indices = defaultdict(list)
        for index, (_, date_drive, _) in enumerate(keys):
            seq2indices[date_drive].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(
                self.data[stage], indices, names=self.image_paths[stage], **kwargs
            )
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
        seq_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(seq_keys))
            seq_keys = [seq_keys[i] for i in perm]
        key_indices = [i for key in seq_keys for i in chunk2idx[key]]
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
        return loader, seq_keys, chunk2idx

class KittiCloudyDataModule(pl.LightningDataModule):
    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "kitti_cloudy",
        # paths and fetch
        "data_dir": DATASETS_PATH / "weather_noise/cloudy",
        "tiles_filename": "tiles.pkl",
        "splits": {
            "train": "train_files.txt",
            "val": "test1_files.txt",
            "test": "test2_files.txt",
        },
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 1, "num_workers": 0},
        },
        "max_num_val": 500,
        "selection_subset_val": "furthest",
        "drop_train_too_close_to_val": 5.0,
        "skip_frames": 1,
        "camera_index": 2,
        # overwrite
        "crop_size_meters": 64,
        "max_init_error": 20,
        #"max_init_error_rotation": 10,
        "max_init_error_rotation_eval": 10,
        "add_map_mask": True,
        "mask_pad": 2,
        "target_focal_length": 256,
    }
    dummy_scene_name = "kitti_cloudy"

    def __init__(self, cfg, tile_manager: Optional[TileManager] = None):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.tile_manager = tile_manager
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")
        assert self.cfg.selection_subset_val in ["random", "furthest"]
        self.splits = {}
        self.shifts = {}
        self.calibrations = {}
        self.data = {}
        self.image_paths = {}

    def prepare_data(self):
        if not (self.root.exists() and (self.root / ".downloaded").exists()):
            raise FileNotFoundError(
                "Cannot find the KITTI dataset, run maploc.data.kitti.prepare"
            )

    def parse_split(self, split_arg):
        if isinstance(split_arg, str):
            names, shifts = parse_split_file(self.root / split_arg)
        elif isinstance(split_arg, collections.abc.Sequence):
            names = []
            shifts = None
            for date_drive in split_arg:
                data_dir = (
                    self.root / date_drive / f"image_{self.cfg.camera_index:02}/data"
                )
                assert data_dir.exists(), data_dir
                date_drive = tuple(date_drive.split("/"))
                n = sorted(date_drive + (p.name,) for p in data_dir.glob("*.png"))
                names.extend(n[:: self.cfg.skip_frames])
        else:
            raise ValueError(split_arg)
        return names, shifts

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            stages = ["train", "val"]
        elif stage is None:
            stages = ["train", "val", "test"]
        else:
            stages = [stage]
        for stage in stages:
            self.splits[stage], self.shifts[stage] = self.parse_split(
                self.cfg.splits[stage]
            )
        do_val_subset = "val" in stages and self.cfg.max_num_val is not None
        if do_val_subset and self.cfg.selection_subset_val == "random":
            select = np.random.RandomState(self.cfg.seed).choice(
                len(self.splits["val"]), self.cfg.max_num_val, replace=False
            )
            self.splits["val"] = [self.splits["val"][i] for i in select]
            if self.shifts["val"] is not None:
                self.shifts["val"] = self.shifts["val"][select]
        dates = {d for ns in self.splits.values() for d, _, _ in ns}
        for d in dates:
            self.calibrations[d] = get_camera_calibration(
                self.root / d, self.cfg.camera_index
            )
        if self.tile_manager is None:
            logger.info("Loading the tile manager...")
            self.tile_manager = TileManager.load(self.root / self.cfg.tiles_filename)
        self.cfg.num_classes = {k: len(g) for k, g in self.tile_manager.groups.items()}
        self.cfg.pixel_per_meter = self.tile_manager.ppm

        # pack all attributes in a single tensor to optimize memory access
        self.pack_data(stages)

        dists = None
        if do_val_subset and self.cfg.selection_subset_val == "furthest":
            dists = torch.cdist(
                self.data["val"]["t_c2w"][:, :2].double(),
                self.data["train"]["t_c2w"][:, :2].double(),
            )
            min_dists = dists.min(1).values
            select = torch.argsort(min_dists)[-self.cfg.max_num_val :]
            dists = dists[select]
            self.splits["val"] = [self.splits["val"][i] for i in select]
            if self.shifts["val"] is not None:
                self.shifts["val"] = self.shifts["val"][select]
            for k in list(self.data["val"]):
                if k != "cameras":
                    self.data["val"][k] = self.data["val"][k][select]
            self.image_paths["val"] = self.image_paths["val"][select]

        if "train" in stages and self.cfg.drop_train_too_close_to_val is not None:
            if dists is None:
                dists = torch.cdist(
                    self.data["val"]["t_c2w"][:, :2].double(),
                    self.data["train"]["t_c2w"][:, :2].double(),
                )
            drop = torch.any(dists < self.cfg.drop_train_too_close_to_val, 0)
            select = torch.where(~drop)[0]
            logger.info(
                "Dropping %d (%f %%) images that are too close to validation images.",
                drop.sum(),
                drop.float().mean(),
            )
            self.splits["train"] = [self.splits["train"][i] for i in select]
            if self.shifts["train"] is not None:
                self.shifts["train"] = self.shifts["train"][select]
            for k in list(self.data["train"]):
                if k != "cameras":
                    self.data["train"][k] = self.data["train"][k][select]
            self.image_paths["train"] = self.image_paths["train"][select]

    def pack_data(self, stages):
        for stage in stages:
            names = []
            data = {}
            for i, (date, drive, index) in enumerate(self.splits[stage]):
                d = self.get_frame_data(date, drive, index)
                for k, v in d.items():
                    if i == 0:
                        data[k] = []
                    data[k].append(v)
                path = f"{date}/{drive}/image_{self.cfg.camera_index:02}/data/{index}"
                names.append((self.dummy_scene_name, f"{date}/{drive}", path))
            for k in list(data):
                data[k] = torch.from_numpy(np.stack(data[k]))
            data["camera_id"] = np.full(len(names), self.cfg.camera_index)

            sequences = {date_drive for _, date_drive, _ in names}
            data["cameras"] = {
                self.dummy_scene_name: {
                    seq: {
                        self.cfg.camera_index: self.calibrations[seq.split("/")[0]][0]
                    }
                    for seq in sequences
                }
            }
            shifts = self.shifts[stage]
            if shifts is not None:
                data["shifts"] = torch.from_numpy(shifts.astype(np.float32))
            self.data[stage] = data
            self.image_paths[stage] = np.array(names)

    def get_frame_data(self, date, drive, index):
        _, R_cam_gps, t_cam_gps = self.calibrations[date]

        # Transform the GPS pose to the camera pose
        gps_path = (
            self.root / date / drive / "oxts/data" / Path(index).with_suffix(".txt")
        )
        _, R_world_gps, t_world_gps = parse_gps_file(
            gps_path, self.tile_manager.projection
        )
        R_world_cam = R_world_gps @ R_cam_gps.T
        t_world_cam = t_world_gps - R_world_gps @ R_cam_gps.T @ t_cam_gps
        # Some voodoo to extract correct Euler angles from R_world_cam
        R_cv_xyz = Rotation.from_euler("YX", [-90, 90], degrees=True).as_matrix()
        R_world_cam_xyz = R_world_cam @ R_cv_xyz
        y, p, r = Rotation.from_matrix(R_world_cam_xyz).as_euler("ZYX", degrees=True)
        roll, pitch, yaw = r, -p, 90 - y
        roll_pitch_yaw = np.array([-roll, -pitch, yaw], np.float32)  # for some reason

        return {
            "t_c2w": t_world_cam.astype(np.float32),
            "roll_pitch_yaw": roll_pitch_yaw,
            "index": int(index.split(".")[0]),
        }

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.image_paths[stage],
            self.data[stage],
            {self.dummy_scene_name: self.root},
            {self.dummy_scene_name: self.tile_manager},
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
        keys = self.image_paths[stage]
        # group images by sequence (date/drive)
        seq2indices = defaultdict(list)
        for index, (_, date_drive, _) in enumerate(keys):
            seq2indices[date_drive].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(
                self.data[stage], indices, names=self.image_paths[stage], **kwargs
            )
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
        seq_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(seq_keys))
            seq_keys = [seq_keys[i] for i in perm]
        key_indices = [i for key in seq_keys for i in chunk2idx[key]]
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
        return loader, seq_keys, chunk2id
    
class KittiMotionBlurDataModule(pl.LightningDataModule):
    
    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "kitti_motion_blur",
        # paths and fetch
        "data_dir": DATASETS_PATH / "sensor_noise/motion_blur",
        "tiles_filename": "tiles.pkl",
        "splits": {
            "train": "train_files.txt",
            "val": "test1_files.txt",
            "test": "test2_files.txt",
        },
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 1, "num_workers": 0},
        },
        "max_num_val": 500,
        "selection_subset_val": "furthest",
        "drop_train_too_close_to_val": 5.0,
        "skip_frames": 1,
        "camera_index": 2,
        # overwrite
        "crop_size_meters": 64,
        "max_init_error": 20,
        #"max_init_error_rotation": 10,
        "max_init_error_rotation_eval": 10,
        "add_map_mask": True,
        "mask_pad": 2,
        "target_focal_length": 256,
    }
    dummy_scene_name = "kitti_motion_blur"

    def __init__(self, cfg, tile_manager: Optional[TileManager] = None):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.tile_manager = tile_manager
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")
        assert self.cfg.selection_subset_val in ["random", "furthest"]
        self.splits = {}
        self.shifts = {}
        self.calibrations = {}
        self.data = {}
        self.image_paths = {}

    def prepare_data(self):
        if not (self.root.exists() and (self.root / ".downloaded").exists()):
            raise FileNotFoundError(
                "Cannot find the KITTI dataset, run maploc.data.kitti.prepare"
            )

    def parse_split(self, split_arg):
        if isinstance(split_arg, str):
            names, shifts = parse_split_file(self.root / split_arg)
        elif isinstance(split_arg, collections.abc.Sequence):
            names = []
            shifts = None
            for date_drive in split_arg:
                data_dir = (
                    self.root / date_drive / f"image_{self.cfg.camera_index:02}/data"
                )
                assert data_dir.exists(), data_dir
                date_drive = tuple(date_drive.split("/"))
                n = sorted(date_drive + (p.name,) for p in data_dir.glob("*.png"))
                names.extend(n[:: self.cfg.skip_frames])
        else:
            raise ValueError(split_arg)
        return names, shifts

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            stages = ["train", "val"]
        elif stage is None:
            stages = ["train", "val", "test"]
        else:
            stages = [stage]
        for stage in stages:
            self.splits[stage], self.shifts[stage] = self.parse_split(
                self.cfg.splits[stage]
            )
        do_val_subset = "val" in stages and self.cfg.max_num_val is not None
        if do_val_subset and self.cfg.selection_subset_val == "random":
            select = np.random.RandomState(self.cfg.seed).choice(
                len(self.splits["val"]), self.cfg.max_num_val, replace=False
            )
            self.splits["val"] = [self.splits["val"][i] for i in select]
            if self.shifts["val"] is not None:
                self.shifts["val"] = self.shifts["val"][select]
        dates = {d for ns in self.splits.values() for d, _, _ in ns}
        for d in dates:
            self.calibrations[d] = get_camera_calibration(
                self.root / d, self.cfg.camera_index
            )
        if self.tile_manager is None:
            logger.info("Loading the tile manager...")
            self.tile_manager = TileManager.load(self.root / self.cfg.tiles_filename)
        self.cfg.num_classes = {k: len(g) for k, g in self.tile_manager.groups.items()}
        self.cfg.pixel_per_meter = self.tile_manager.ppm

        # pack all attributes in a single tensor to optimize memory access
        self.pack_data(stages)

        dists = None
        if do_val_subset and self.cfg.selection_subset_val == "furthest":
            dists = torch.cdist(
                self.data["val"]["t_c2w"][:, :2].double(),
                self.data["train"]["t_c2w"][:, :2].double(),
            )
            min_dists = dists.min(1).values
            select = torch.argsort(min_dists)[-self.cfg.max_num_val :]
            dists = dists[select]
            self.splits["val"] = [self.splits["val"][i] for i in select]
            if self.shifts["val"] is not None:
                self.shifts["val"] = self.shifts["val"][select]
            for k in list(self.data["val"]):
                if k != "cameras":
                    self.data["val"][k] = self.data["val"][k][select]
            self.image_paths["val"] = self.image_paths["val"][select]

        if "train" in stages and self.cfg.drop_train_too_close_to_val is not None:
            if dists is None:
                dists = torch.cdist(
                    self.data["val"]["t_c2w"][:, :2].double(),
                    self.data["train"]["t_c2w"][:, :2].double(),
                )
            drop = torch.any(dists < self.cfg.drop_train_too_close_to_val, 0)
            select = torch.where(~drop)[0]
            logger.info(
                "Dropping %d (%f %%) images that are too close to validation images.",
                drop.sum(),
                drop.float().mean(),
            )
            self.splits["train"] = [self.splits["train"][i] for i in select]
            if self.shifts["train"] is not None:
                self.shifts["train"] = self.shifts["train"][select]
            for k in list(self.data["train"]):
                if k != "cameras":
                    self.data["train"][k] = self.data["train"][k][select]
            self.image_paths["train"] = self.image_paths["train"][select]

    def pack_data(self, stages):
        for stage in stages:
            names = []
            data = {}
            for i, (date, drive, index) in enumerate(self.splits[stage]):
                d = self.get_frame_data(date, drive, index)
                for k, v in d.items():
                    if i == 0:
                        data[k] = []
                    data[k].append(v)
                path = f"{date}/{drive}/image_{self.cfg.camera_index:02}/data/{index}"
                names.append((self.dummy_scene_name, f"{date}/{drive}", path))
            for k in list(data):
                data[k] = torch.from_numpy(np.stack(data[k]))
            data["camera_id"] = np.full(len(names), self.cfg.camera_index)

            sequences = {date_drive for _, date_drive, _ in names}
            data["cameras"] = {
                self.dummy_scene_name: {
                    seq: {
                        self.cfg.camera_index: self.calibrations[seq.split("/")[0]][0]
                    }
                    for seq in sequences
                }
            }
            shifts = self.shifts[stage]
            if shifts is not None:
                data["shifts"] = torch.from_numpy(shifts.astype(np.float32))
            self.data[stage] = data
            self.image_paths[stage] = np.array(names)

    def get_frame_data(self, date, drive, index):
        _, R_cam_gps, t_cam_gps = self.calibrations[date]

        # Transform the GPS pose to the camera pose
        gps_path = (
            self.root / date / drive / "oxts/data" / Path(index).with_suffix(".txt")
        )
        _, R_world_gps, t_world_gps = parse_gps_file(
            gps_path, self.tile_manager.projection
        )
        R_world_cam = R_world_gps @ R_cam_gps.T
        t_world_cam = t_world_gps - R_world_gps @ R_cam_gps.T @ t_cam_gps
        # Some voodoo to extract correct Euler angles from R_world_cam
        R_cv_xyz = Rotation.from_euler("YX", [-90, 90], degrees=True).as_matrix()
        R_world_cam_xyz = R_world_cam @ R_cv_xyz
        y, p, r = Rotation.from_matrix(R_world_cam_xyz).as_euler("ZYX", degrees=True)
        roll, pitch, yaw = r, -p, 90 - y
        roll_pitch_yaw = np.array([-roll, -pitch, yaw], np.float32)  # for some reason

        return {
            "t_c2w": t_world_cam.astype(np.float32),
            "roll_pitch_yaw": roll_pitch_yaw,
            "index": int(index.split(".")[0]),
        }

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.image_paths[stage],
            self.data[stage],
            {self.dummy_scene_name: self.root},
            {self.dummy_scene_name: self.tile_manager},
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
        keys = self.image_paths[stage]
        # group images by sequence (date/drive)
        seq2indices = defaultdict(list)
        for index, (_, date_drive, _) in enumerate(keys):
            seq2indices[date_drive].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(
                self.data[stage], indices, names=self.image_paths[stage], **kwargs
            )
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
        seq_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(seq_keys))
            seq_keys = [seq_keys[i] for i in perm]
        key_indices = [i for key in seq_keys for i in chunk2idx[key]]
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
        return loader, seq_keys, chunk2id
    
class KittiUnderExposureDataModule(pl.LightningDataModule):
    
    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "kitti_under",
        # paths and fetch
        "data_dir": DATASETS_PATH / "sensor_noise/under_exposure",
        "tiles_filename": "tiles.pkl",
        "splits": {
            "train": "train_files.txt",
            "val": "test1_files.txt",
            "test": "test2_files.txt",
        },
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 1, "num_workers": 0},
        },
        "max_num_val": 500,
        "selection_subset_val": "furthest",
        "drop_train_too_close_to_val": 5.0,
        "skip_frames": 1,
        "camera_index": 2,
        # overwrite
        "crop_size_meters": 64,
        "max_init_error": 20,
        #"max_init_error_rotation": 10,
        "max_init_error_rotation_eval": 10,
        "add_map_mask": True,
        "mask_pad": 2,
        "target_focal_length": 256,
    }
    dummy_scene_name = "kitti_under"

    def __init__(self, cfg, tile_manager: Optional[TileManager] = None):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.tile_manager = tile_manager
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")
        assert self.cfg.selection_subset_val in ["random", "furthest"]
        self.splits = {}
        self.shifts = {}
        self.calibrations = {}
        self.data = {}
        self.image_paths = {}

    def prepare_data(self):
        if not (self.root.exists() and (self.root / ".downloaded").exists()):
            raise FileNotFoundError(
                "Cannot find the KITTI dataset, run maploc.data.kitti.prepare"
            )

    def parse_split(self, split_arg):
        if isinstance(split_arg, str):
            names, shifts = parse_split_file(self.root / split_arg)
        elif isinstance(split_arg, collections.abc.Sequence):
            names = []
            shifts = None
            for date_drive in split_arg:
                data_dir = (
                    self.root / date_drive / f"image_{self.cfg.camera_index:02}/data"
                )
                assert data_dir.exists(), data_dir
                date_drive = tuple(date_drive.split("/"))
                n = sorted(date_drive + (p.name,) for p in data_dir.glob("*.png"))
                names.extend(n[:: self.cfg.skip_frames])
        else:
            raise ValueError(split_arg)
        return names, shifts

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            stages = ["train", "val"]
        elif stage is None:
            stages = ["train", "val", "test"]
        else:
            stages = [stage]
        for stage in stages:
            self.splits[stage], self.shifts[stage] = self.parse_split(
                self.cfg.splits[stage]
            )
        do_val_subset = "val" in stages and self.cfg.max_num_val is not None
        if do_val_subset and self.cfg.selection_subset_val == "random":
            select = np.random.RandomState(self.cfg.seed).choice(
                len(self.splits["val"]), self.cfg.max_num_val, replace=False
            )
            self.splits["val"] = [self.splits["val"][i] for i in select]
            if self.shifts["val"] is not None:
                self.shifts["val"] = self.shifts["val"][select]
        dates = {d for ns in self.splits.values() for d, _, _ in ns}
        for d in dates:
            self.calibrations[d] = get_camera_calibration(
                self.root / d, self.cfg.camera_index
            )
        if self.tile_manager is None:
            logger.info("Loading the tile manager...")
            self.tile_manager = TileManager.load(self.root / self.cfg.tiles_filename)
        self.cfg.num_classes = {k: len(g) for k, g in self.tile_manager.groups.items()}
        self.cfg.pixel_per_meter = self.tile_manager.ppm

        # pack all attributes in a single tensor to optimize memory access
        self.pack_data(stages)

        dists = None
        if do_val_subset and self.cfg.selection_subset_val == "furthest":
            dists = torch.cdist(
                self.data["val"]["t_c2w"][:, :2].double(),
                self.data["train"]["t_c2w"][:, :2].double(),
            )
            min_dists = dists.min(1).values
            select = torch.argsort(min_dists)[-self.cfg.max_num_val :]
            dists = dists[select]
            self.splits["val"] = [self.splits["val"][i] for i in select]
            if self.shifts["val"] is not None:
                self.shifts["val"] = self.shifts["val"][select]
            for k in list(self.data["val"]):
                if k != "cameras":
                    self.data["val"][k] = self.data["val"][k][select]
            self.image_paths["val"] = self.image_paths["val"][select]

        if "train" in stages and self.cfg.drop_train_too_close_to_val is not None:
            if dists is None:
                dists = torch.cdist(
                    self.data["val"]["t_c2w"][:, :2].double(),
                    self.data["train"]["t_c2w"][:, :2].double(),
                )
            drop = torch.any(dists < self.cfg.drop_train_too_close_to_val, 0)
            select = torch.where(~drop)[0]
            logger.info(
                "Dropping %d (%f %%) images that are too close to validation images.",
                drop.sum(),
                drop.float().mean(),
            )
            self.splits["train"] = [self.splits["train"][i] for i in select]
            if self.shifts["train"] is not None:
                self.shifts["train"] = self.shifts["train"][select]
            for k in list(self.data["train"]):
                if k != "cameras":
                    self.data["train"][k] = self.data["train"][k][select]
            self.image_paths["train"] = self.image_paths["train"][select]

    def pack_data(self, stages):
        for stage in stages:
            names = []
            data = {}
            for i, (date, drive, index) in enumerate(self.splits[stage]):
                d = self.get_frame_data(date, drive, index)
                for k, v in d.items():
                    if i == 0:
                        data[k] = []
                    data[k].append(v)
                path = f"{date}/{drive}/image_{self.cfg.camera_index:02}/data/{index}"
                names.append((self.dummy_scene_name, f"{date}/{drive}", path))
            for k in list(data):
                data[k] = torch.from_numpy(np.stack(data[k]))
            data["camera_id"] = np.full(len(names), self.cfg.camera_index)

            sequences = {date_drive for _, date_drive, _ in names}
            data["cameras"] = {
                self.dummy_scene_name: {
                    seq: {
                        self.cfg.camera_index: self.calibrations[seq.split("/")[0]][0]
                    }
                    for seq in sequences
                }
            }
            shifts = self.shifts[stage]
            if shifts is not None:
                data["shifts"] = torch.from_numpy(shifts.astype(np.float32))
            self.data[stage] = data
            self.image_paths[stage] = np.array(names)

    def get_frame_data(self, date, drive, index):
        _, R_cam_gps, t_cam_gps = self.calibrations[date]

        # Transform the GPS pose to the camera pose
        gps_path = (
            self.root / date / drive / "oxts/data" / Path(index).with_suffix(".txt")
        )
        _, R_world_gps, t_world_gps = parse_gps_file(
            gps_path, self.tile_manager.projection
        )
        R_world_cam = R_world_gps @ R_cam_gps.T
        t_world_cam = t_world_gps - R_world_gps @ R_cam_gps.T @ t_cam_gps
        # Some voodoo to extract correct Euler angles from R_world_cam
        R_cv_xyz = Rotation.from_euler("YX", [-90, 90], degrees=True).as_matrix()
        R_world_cam_xyz = R_world_cam @ R_cv_xyz
        y, p, r = Rotation.from_matrix(R_world_cam_xyz).as_euler("ZYX", degrees=True)
        roll, pitch, yaw = r, -p, 90 - y
        roll_pitch_yaw = np.array([-roll, -pitch, yaw], np.float32)  # for some reason

        return {
            "t_c2w": t_world_cam.astype(np.float32),
            "roll_pitch_yaw": roll_pitch_yaw,
            "index": int(index.split(".")[0]),
        }

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.image_paths[stage],
            self.data[stage],
            {self.dummy_scene_name: self.root},
            {self.dummy_scene_name: self.tile_manager},
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
        keys = self.image_paths[stage]
        # group images by sequence (date/drive)
        seq2indices = defaultdict(list)
        for index, (_, date_drive, _) in enumerate(keys):
            seq2indices[date_drive].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(
                self.data[stage], indices, names=self.image_paths[stage], **kwargs
            )
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
        seq_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(seq_keys))
            seq_keys = [seq_keys[i] for i in perm]
        key_indices = [i for key in seq_keys for i in chunk2idx[key]]
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
        return loader, seq_keys, chunk2id

class KittiOverExposureDataModule(pl.LightningDataModule):
    
    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "kitti_over",
        # paths and fetch
        "data_dir": DATASETS_PATH / "sensor_noise/over_exposure",
        "tiles_filename": "tiles.pkl",
        "splits": {
            "train": "train_files.txt",
            "val": "test1_files.txt",
            "test": "test2_files.txt",
        },
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 1, "num_workers": 0},
        },
        "max_num_val": 500,
        "selection_subset_val": "furthest",
        "drop_train_too_close_to_val": 5.0,
        "skip_frames": 1,
        "camera_index": 2,
        # overwrite
        "crop_size_meters": 64,
        "max_init_error": 20,
        #"max_init_error_rotation": 10,
        "max_init_error_rotation_eval": 10,
        "add_map_mask": True,
        "mask_pad": 2,
        "target_focal_length": 256,
    }
    dummy_scene_name = "kitti_over"

    def __init__(self, cfg, tile_manager: Optional[TileManager] = None):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.tile_manager = tile_manager
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")
        assert self.cfg.selection_subset_val in ["random", "furthest"]
        self.splits = {}
        self.shifts = {}
        self.calibrations = {}
        self.data = {}
        self.image_paths = {}

    def prepare_data(self):
        if not (self.root.exists() and (self.root / ".downloaded").exists()):
            raise FileNotFoundError(
                "Cannot find the KITTI dataset, run maploc.data.kitti.prepare"
            )

    def parse_split(self, split_arg):
        if isinstance(split_arg, str):
            names, shifts = parse_split_file(self.root / split_arg)
        elif isinstance(split_arg, collections.abc.Sequence):
            names = []
            shifts = None
            for date_drive in split_arg:
                data_dir = (
                    self.root / date_drive / f"image_{self.cfg.camera_index:02}/data"
                )
                assert data_dir.exists(), data_dir
                date_drive = tuple(date_drive.split("/"))
                n = sorted(date_drive + (p.name,) for p in data_dir.glob("*.png"))
                names.extend(n[:: self.cfg.skip_frames])
        else:
            raise ValueError(split_arg)
        return names, shifts

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            stages = ["train", "val"]
        elif stage is None:
            stages = ["train", "val", "test"]
        else:
            stages = [stage]
        for stage in stages:
            self.splits[stage], self.shifts[stage] = self.parse_split(
                self.cfg.splits[stage]
            )
        do_val_subset = "val" in stages and self.cfg.max_num_val is not None
        if do_val_subset and self.cfg.selection_subset_val == "random":
            select = np.random.RandomState(self.cfg.seed).choice(
                len(self.splits["val"]), self.cfg.max_num_val, replace=False
            )
            self.splits["val"] = [self.splits["val"][i] for i in select]
            if self.shifts["val"] is not None:
                self.shifts["val"] = self.shifts["val"][select]
        dates = {d for ns in self.splits.values() for d, _, _ in ns}
        for d in dates:
            self.calibrations[d] = get_camera_calibration(
                self.root / d, self.cfg.camera_index
            )
        if self.tile_manager is None:
            logger.info("Loading the tile manager...")
            self.tile_manager = TileManager.load(self.root / self.cfg.tiles_filename)
        self.cfg.num_classes = {k: len(g) for k, g in self.tile_manager.groups.items()}
        self.cfg.pixel_per_meter = self.tile_manager.ppm

        # pack all attributes in a single tensor to optimize memory access
        self.pack_data(stages)

        dists = None
        if do_val_subset and self.cfg.selection_subset_val == "furthest":
            dists = torch.cdist(
                self.data["val"]["t_c2w"][:, :2].double(),
                self.data["train"]["t_c2w"][:, :2].double(),
            )
            min_dists = dists.min(1).values
            select = torch.argsort(min_dists)[-self.cfg.max_num_val :]
            dists = dists[select]
            self.splits["val"] = [self.splits["val"][i] for i in select]
            if self.shifts["val"] is not None:
                self.shifts["val"] = self.shifts["val"][select]
            for k in list(self.data["val"]):
                if k != "cameras":
                    self.data["val"][k] = self.data["val"][k][select]
            self.image_paths["val"] = self.image_paths["val"][select]

        if "train" in stages and self.cfg.drop_train_too_close_to_val is not None:
            if dists is None:
                dists = torch.cdist(
                    self.data["val"]["t_c2w"][:, :2].double(),
                    self.data["train"]["t_c2w"][:, :2].double(),
                )
            drop = torch.any(dists < self.cfg.drop_train_too_close_to_val, 0)
            select = torch.where(~drop)[0]
            logger.info(
                "Dropping %d (%f %%) images that are too close to validation images.",
                drop.sum(),
                drop.float().mean(),
            )
            self.splits["train"] = [self.splits["train"][i] for i in select]
            if self.shifts["train"] is not None:
                self.shifts["train"] = self.shifts["train"][select]
            for k in list(self.data["train"]):
                if k != "cameras":
                    self.data["train"][k] = self.data["train"][k][select]
            self.image_paths["train"] = self.image_paths["train"][select]

    def pack_data(self, stages):
        for stage in stages:
            names = []
            data = {}
            for i, (date, drive, index) in enumerate(self.splits[stage]):
                d = self.get_frame_data(date, drive, index)
                for k, v in d.items():
                    if i == 0:
                        data[k] = []
                    data[k].append(v)
                path = f"{date}/{drive}/image_{self.cfg.camera_index:02}/data/{index}"
                names.append((self.dummy_scene_name, f"{date}/{drive}", path))
            for k in list(data):
                data[k] = torch.from_numpy(np.stack(data[k]))
            data["camera_id"] = np.full(len(names), self.cfg.camera_index)

            sequences = {date_drive for _, date_drive, _ in names}
            data["cameras"] = {
                self.dummy_scene_name: {
                    seq: {
                        self.cfg.camera_index: self.calibrations[seq.split("/")[0]][0]
                    }
                    for seq in sequences
                }
            }
            shifts = self.shifts[stage]
            if shifts is not None:
                data["shifts"] = torch.from_numpy(shifts.astype(np.float32))
            self.data[stage] = data
            self.image_paths[stage] = np.array(names)

    def get_frame_data(self, date, drive, index):
        _, R_cam_gps, t_cam_gps = self.calibrations[date]

        # Transform the GPS pose to the camera pose
        gps_path = (
            self.root / date / drive / "oxts/data" / Path(index).with_suffix(".txt")
        )
        _, R_world_gps, t_world_gps = parse_gps_file(
            gps_path, self.tile_manager.projection
        )
        R_world_cam = R_world_gps @ R_cam_gps.T
        t_world_cam = t_world_gps - R_world_gps @ R_cam_gps.T @ t_cam_gps
        # Some voodoo to extract correct Euler angles from R_world_cam
        R_cv_xyz = Rotation.from_euler("YX", [-90, 90], degrees=True).as_matrix()
        R_world_cam_xyz = R_world_cam @ R_cv_xyz
        y, p, r = Rotation.from_matrix(R_world_cam_xyz).as_euler("ZYX", degrees=True)
        roll, pitch, yaw = r, -p, 90 - y
        roll_pitch_yaw = np.array([-roll, -pitch, yaw], np.float32)  # for some reason

        return {
            "t_c2w": t_world_cam.astype(np.float32),
            "roll_pitch_yaw": roll_pitch_yaw,
            "index": int(index.split(".")[0]),
        }

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.image_paths[stage],
            self.data[stage],
            {self.dummy_scene_name: self.root},
            {self.dummy_scene_name: self.tile_manager},
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
        keys = self.image_paths[stage]
        # group images by sequence (date/drive)
        seq2indices = defaultdict(list)
        for index, (_, date_drive, _) in enumerate(keys):
            seq2indices[date_drive].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(
                self.data[stage], indices, names=self.image_paths[stage], **kwargs
            )
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
        seq_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(seq_keys))
            seq_keys = [seq_keys[i] for i in perm]
        key_indices = [i for key in seq_keys for i in chunk2idx[key]]
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
        return loader, seq_keys, chunk2id