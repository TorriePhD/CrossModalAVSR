from operator import itemgetter
from typing import Iterator, Optional

import numpy as np
import torch

from fairseq.data import data_utils
from torch.utils.data import Dataset, DistributedSampler, RandomSampler
from torch.utils.data.sampler import Sampler
from math import ceil
import random


class ByFrameCountSampler(Sampler):
    def __init__(self, dataset, max_frames_per_gpu, shuffle=True, seed=0,modality = "audiovisual",validation=False):
        self.dataset = dataset
        self.max_frames_per_gpu = max_frames_per_gpu
        self.sizes = [item[2] for item in self.dataset.list]

        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        self.validation = validation
        self.modality = modality
        if self.modality == "audiovisual" and not self.validation:
            self._get_AV_Batch()
        else:
            batch_indices = data_utils.batch_by_size(
                self._get_indices(), lambda i: self.sizes[i], max_tokens=max_frames_per_gpu
            )
            self.num_batches = len(batch_indices)

    def slight_shuffle(self, sorted_list, window_ratio=0.01,AV=False):

        length = len(sorted_list)
        window_size = max(1, int(length * window_ratio))
        
        shuffled_list = sorted_list[:]
        for i in range(length):
            # Define the window for swapping
            start = max(0, i - window_size)
            end = min(length, i + window_size + 1)
            
            # Choose a random index within the window
            swap_with = random.randint(start, end - 1)
            
            # Swap elements
            shuffled_list[i], shuffled_list[swap_with] = shuffled_list[swap_with], shuffled_list[i]
        
        return shuffled_list
    def _get_AV_Batch(self):
        self.audio_only_indices = []
        random.seed(self.seed + self.epoch)
        # self.video_only_indices = []
        self.audiovisual_indices = []
        for i in range(len(self.dataset.list)):
            if self.dataset.list[i][-1] == "audio":
                self.audio_only_indices.append(i)
            # elif self.dataset.list[i][-1] == "video":
            #     self.video_only_indices.append(i)
            else:
                self.audiovisual_indices.append(i)
        # Shuffle indices if required
        self.audiovisual_indices.sort(key=lambda idx: self.sizes[idx], reverse=True)
        self.audio_only_indices.sort(key=lambda idx: self.sizes[idx] // 4, reverse=True)
        if self.shuffle:
            self.audiovisual_indices = self.slight_shuffle(self.audiovisual_indices, window_ratio=0.001,AV=True)
            self.audio_only_indices = self.slight_shuffle(self.audio_only_indices, window_ratio=0.001)
        # Iterate and yield balanced batches
        batch = []
        length = 0
        self.batches = []
        # while self.audio_only_indices or self.video_only_indices or self.audiovisual_indices:
        audioBatchCount = 0
        audioVisualBatchCount = 0
        copyAudioVisualIndices = self.audiovisual_indices.copy()
        #shuffle
        random.shuffle(copyAudioVisualIndices)
        batchChanged = False
        ogRatio = len(self.audiovisual_indices) / (len(self.audio_only_indices)) if len(self.audio_only_indices) != 0 else 0

        while self.audio_only_indices or self.audiovisual_indices:
            
            fullRatio = len(self.audiovisual_indices) / (len(self.audio_only_indices)) if len(self.audio_only_indices) != 0 else 0
            batchRatio = audioVisualBatchCount / audioBatchCount if audioBatchCount != 0 else 0 
            if self.audiovisual_indices:
                nextAudioVisualIndicie = self.audiovisual_indices[-1]
                size = self.sizes[nextAudioVisualIndicie]
                if (size+length)<self.max_frames_per_gpu and  self.audiovisual_indices and ((fullRatio > batchRatio or ogRatio>1) or audioVisualBatchCount == 0 or len(self.audio_only_indices) == 0):
                    batch.append(self.audiovisual_indices.pop())
                    length += self.sizes[batch[-1]]
                    audioVisualBatchCount += 1
                    batchChanged = True
            if len(self.audiovisual_indices) == 0 and len(self.audio_only_indices) > 0 and audioVisualBatchCount == 0:
                batch.append(copyAudioVisualIndices.pop())
                length += self.sizes[batch[-1]]
                audioVisualBatchCount += 1
                batchChanged = True
            
            if (not self.audio_only_indices and not self.audiovisual_indices) or length >= self.max_frames_per_gpu:
                self.batches.append(batch)
                batch = []
                length = 0
                audioBatchCount = 0
                audioVisualBatchCount = 0
                if len(self.audio_only_indices) == 0 and len(self.audiovisual_indices) == 0:
                    break
            batchRatio = audioVisualBatchCount / audioBatchCount if audioBatchCount != 0 else 0 #0
            fullRatio = len(self.audiovisual_indices) / len(self.audio_only_indices) if len(self.audio_only_indices) != 0 else 0
            if self.audio_only_indices:
                nextAudioIndicie = self.audio_only_indices[-1]
                size = self.sizes[nextAudioIndicie]//4
                if self.audio_only_indices and ((fullRatio <= batchRatio or ogRatio>1) or audioBatchCount == 0) and (size+length)<self.max_frames_per_gpu:
                    batch.append(self.audio_only_indices.pop())
                    length += self.sizes[batch[-1]]//4
                    audioBatchCount += 1
                    batchChanged = True
            # if self.video_only_indices:
            #     batch.append(self.video_only_indices.pop())
            #     length += self.sizes[batch[-1]]
            

            # if (not self.audio_only_indices and not self.video_only_indices and not self.audiovisual_indices) or length >= self.max_frames_per_gpu:
            if (not self.audio_only_indices and not self.audiovisual_indices) or length >= self.max_frames_per_gpu or not batchChanged:
                self.batches.append(batch)
                batch = []
                length = 0
                audioBatchCount = 0
                audioVisualBatchCount = 0
            batchChanged = False
        if self.shuffle:
            random.seed(self.seed + self.epoch)
            random.shuffle(self.batches)
        print("Batch count: ",len(self.batches))
        self.batchCount = len(self.batches)
    def _get_indices(self):
        if self.shuffle:  # shuffles indices corresponding to equal lengths
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            order = [torch.randperm(len(self.dataset), generator=g).tolist()]

        else:
            order = [list(range(len(self.dataset)))]
        order.append(self.sizes)
        return np.lexsort(order)[::-1]

    def __len__(self):
        if self.modality == "audiovisual" and not self.validation:
            return self.batchCount
        return self.num_batches

    def __iter__(self):
        if self.modality == "audiovisual" and not self.validation:
            self._get_AV_Batch()
            return iter(self.batches)
        else:
            batch_indices = data_utils.batch_by_size(
                self._get_indices(),
                lambda i: self.sizes[i],
                max_tokens=self.max_frames_per_gpu,
            )
            return iter(batch_indices)

    def set_epoch(self, epoch):
        self.epoch = epoch


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        if len(self.sampler_list) <= index:
            print("Index out of range", index, len(self.sampler_list))
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()

        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.sampler.set_epoch(epoch)


class RandomSamplerWrapper(RandomSampler):
    def __init__(self, sampler):
        super(RandomSamplerWrapper, self).__init__(DatasetFromSampler(sampler))
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))
