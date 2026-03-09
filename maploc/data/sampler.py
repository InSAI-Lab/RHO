import torch
from torch.utils.data import Sampler, BatchSampler
import numpy as np

class MinDistanceBatchSampler(BatchSampler):
    """
    A batch sampler that ensures all anchors in a batch are at least a minimum
    distance apart from each other. Inherits from BatchSampler for full compatibility
    with PyTorch Lightning's distributed training.

    Args:
        sampler (Sampler): The base sampler that provides the stream of indices.
                           In a distributed setting, this will be a DistributedSampler.
        positions (torch.Tensor): A tensor of shape [N, 2] or [N, 3] containing
                                  the 2D or 3D coordinates of each sample.
        batch_size (int): The desired number of samples per batch.
        drop_last (bool): If true, the sampler will drop the last batch if
                          it is smaller than batch_size.
        min_distance (float): The minimum required distance between any two
                              samples in a batch.
    """
    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool, 
                positions: torch.Tensor, min_distance: float):
        # Pass the base sampler, batch_size, and drop_last to the parent constructor.
        super().__init__(sampler, batch_size, drop_last)

        if not isinstance(positions, torch.Tensor):
            self.positions = torch.tensor(positions, dtype=torch.float)

        self.positions = positions
        self.min_distance_sq = min_distance ** 2  # Use squared distance for efficiency

    def __iter__(self):
        # The self.sampler (e.g., RandomSampler or DistributedSampler) gives us the indices.
        # We process these indices to form our distance-constrained batches.
        batch = []
        for idx in self.sampler:
            pos_candidate = self.positions[idx]
            
            # Check if the candidate is far enough from all samples already in the batch.
            is_far_enough = True
            for batch_idx in batch:
                pos_in_batch = self.positions[batch_idx]
                dist_sq = torch.sum((pos_candidate - pos_in_batch) ** 2)
                if dist_sq < self.min_distance_sq:
                    is_far_enough = False
                    break
            
            if is_far_enough:
                batch.append(idx)

            # If the batch is full, yield it and start a new one.
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        # After the loop, yield the last batch if it's not empty and we're not dropping it.
        if len(batch) > 0 and not self.drop_last:
            yield batch
