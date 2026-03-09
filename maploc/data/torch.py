# Copyright (c) Meta Platforms, Inc. and affiliates.

import collections
import os

import torch
import torch.distributed as dist
from typing import Iterator, List, Sized
from lightning_fabric.utilities.apply_func import move_data_to_device
from lightning_fabric.utilities.seed import pl_worker_init_function
from lightning_utilities.core.apply_func import apply_to_collection
from torch.utils.data import get_worker_info, default_collate, Sampler
from torch.utils.data._utils.collate import (
    default_collate_err_msg_format,
    np_str_obj_array_pattern,
)


def collate(batch):
    """Difference with PyTorch default_collate: it can stack other tensor-like objects.
    Adapted from PixLoc, Paul-Edouard Sarlin, ETH Zurich
    https://github.com/cvg/pixloc
    Released under the Apache License 2.0
    """
    if not isinstance(batch, list):  # no batching
        return batch
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel, device=elem.device)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]
    else:
        # try to stack anyway in case the object implements stacking.
        try:
            return torch.stack(batch, 0)
        except TypeError as e:
            if "expected Tensor as element" in str(e):
                return batch
            else:
                raise e


def set_num_threads(nt):
    """Force numpy and other libraries to use a limited number of threads."""
    try:
        import mkl
    except ImportError:
        pass
    else:
        mkl.set_num_threads(nt)
    torch.set_num_threads(1)
    os.environ["IPC_ENABLE"] = "1"
    for o in [
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]:
        os.environ[o] = str(nt)


def worker_init_fn(i):
    info = get_worker_info()
    pl_worker_init_function(info.id)
    num_threads = info.dataset.cfg.get("num_threads")
    if num_threads is not None:
        set_num_threads(num_threads)


def unbatch_to_device(data, device="cpu"):
    data = move_data_to_device(data, device)
    data = apply_to_collection(data, torch.Tensor, lambda x: x.squeeze(0))
    data = apply_to_collection(
        data, list, lambda x: x[0] if len(x) == 1 and isinstance(x[0], str) else x
    )
    return data

def contrastive_collate_fn(batch):
  """
  Enhanced custom collate function that handles the 'positives' key and adds metadata.
  It handles the custom 'positives' list and delegates the rest
  to the project's main `collate` function, which can handle Camera objects.
  """
  # Extract positive images and track counts BEFORE popping
  positive_images = []
  positives_per_anchor = 0
  
  for d in batch:
      pos_list = d['positives']
      positive_images.append(pos_list)
      if positives_per_anchor == 0:  # Set from first item
          positives_per_anchor = len(pos_list)
      elif len(pos_list) != positives_per_anchor:
          raise ValueError(f"Inconsistent number of positives: expected {positives_per_anchor}, got {len(pos_list)}")
  
  # Now pop the positives from batch (as in original code)
  for d in batch:
      d.pop('positives')
  
  # Calculate batch statistics
  num_anchors = len(batch)
  num_positives_total = num_anchors * positives_per_anchor
  
  # Now collate the rest of the data using custom collate (handles Camera objects)
  collated_batch = collate(batch)
  
  # Stack the anchor and positive images into a single tensor
  anchor_images = collated_batch['image']
  flat_positives = [p for pos_list in positive_images for p in pos_list]
  
  # Stack the list of 3D positive tensors into a single 4D tensor
  if flat_positives:
      positive_images_tensor = torch.stack(flat_positives, dim=0)
      # Now both `anchor_images` and `positive_images_tensor` are 4D. Concatenate them.
      all_images = torch.cat([anchor_images, positive_images_tensor], dim=0)
  else:
      all_images = anchor_images
      num_positives_total = 0
  
  # The final 'image' tensor in the batch will contain all anchors followed by all positives
  collated_batch['image'] = all_images
  
  # Add metadata for loss computation
  collated_batch['num_anchors'] = num_anchors
  collated_batch['num_positives'] = num_positives_total
  collated_batch['positives_per_anchor'] = positives_per_anchor
  
  # Add indices for easier extraction
  anchor_indices = list(range(num_anchors))
  positive_indices = list(range(num_anchors, num_anchors + num_positives_total))
  
  collated_batch['anchor_indices'] = torch.tensor(anchor_indices)
  collated_batch['positive_indices'] = torch.tensor(positive_indices)
  
  # Add debug info
  collated_batch['batch_structure'] = {
      'total_samples': num_anchors + num_positives_total,
      'anchor_range': (0, num_anchors),
      'positive_range': (num_anchors, num_anchors + num_positives_total),
      'positives_per_anchor': positives_per_anchor
  }
  
  return collated_batch

class DistributedGroupSampler(Sampler[List[int]]):
    """
    Sampler that restricts data loading to a subset of the dataset,
    while preserving sample groups. It is intended for use with
    :class:`torch.nn.parallel.DistributedDataParallel`.

    It partitions groups of samples (e.g., views of a panorama) among processes,
    and yields batches of indices for these groups.

    Args:
        dataset (Sized): The dataset to sample from.
        samples_per_group (int): The number of samples in each group.
        shuffle (bool, optional): If ``True``, shuffle the groups. Defaults to ``True``.
        seed (int, optional): Random seed used to shuffle the sampler if
            :attr:`shuffle=True`. Defaults to 0.
        drop_last (bool, optional): If ``True``, the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. Defaults to ``False``.
    """

    def __init__(
        self,
        dataset: Sized,
        samples_per_group: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Requires distributed package to be initialized.")

        if len(dataset) % samples_per_group != 0:
            raise ValueError(
                f"Dataset size {len(dataset)} is not divisible by "
                f"samples_per_group {samples_per_group}."
            )

        self.dataset = dataset
        self.num_groups = len(dataset) // samples_per_group
        self.samples_per_group = samples_per_group
        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

        if self.drop_last and self.num_groups % self.num_replicas != 0:
            self.num_samples_per_replica = self.num_groups // self.num_replicas
        else:
            self.num_samples_per_replica = (self.num_groups + self.num_replicas - 1) // self.num_replicas

        self.total_size = self.num_samples_per_replica * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        # 1. Create a list of group indices
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        group_indices = torch.arange(self.num_groups)
        if self.shuffle:
            perm = torch.randperm(self.num_groups, generator=g)
            group_indices = group_indices[perm]

        # 2. Add extra samples to make it evenly divisible (if not dropping last)
        if not self.drop_last:
            padding_size = self.total_size - len(group_indices)
            if padding_size > 0:
                group_indices = torch.cat([group_indices, group_indices[:padding_size]])
        else:
            group_indices = group_indices[:self.total_size]
        
        # 3. Subsample for the current rank
        groups_for_rank = group_indices[self.rank:self.total_size:self.num_replicas]

        # 4. Expand group indices to sample indices
        # [group1_idx, group2_idx] -> [s1, s2, s3, s4, s5, s6]
        # where s1,s2,s3 are from group1 and s4,s5,s6 are from group2
        sample_indices = []
        for group_idx in groups_for_rank:
            start_idx = group_idx * self.samples_per_group
            sample_indices.extend(range(start_idx, start_idx + self.samples_per_group))

        return iter(sample_indices)

    def __len__(self) -> int:
        return self.num_samples_per_replica * self.samples_per_group

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch