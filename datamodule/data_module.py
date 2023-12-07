import os

import torch
from pytorch_lightning import LightningDataModule

from .av_dataset import AVDataset
from .samplers import (
    ByFrameCountSampler,
    DistributedSamplerWrapper,
    RandomSamplerWrapper,
)
from .transforms import AudioTransform, VideoTransform


# https://github.com/facebookresearch/av_hubert/blob/593d0ae8462be128faab6d866a3a926e2955bde1/avhubert/hubert_dataset.py#L517
def pad(samples, pad_val=0.0):
    #if samples[0] is a dictionary call pad on each key
    if isinstance(samples[0], dict):
        return {key: pad([s[key] for s in samples], pad_val) for key in samples[0].keys()}
    lengths = []
    for s in samples:
        if s is not None:
            lengths.append(len(s))
        else:
            lengths.append(0)
    max_size = max(lengths)
    sample_shape = None
    for sample in samples:
        if sample is not None:
            sample_shape = list(sample.shape[1:])
            lenSampleShape = len(sample.shape)
            collated_batch = sample.new_zeros([len(samples), max_size] + sample_shape)
            break
    if sample_shape is None:
        #no samples in batch
        collated_batch = torch.zeros([len(samples), 1,max_size])
        return collated_batch, lengths
    for i, sample in enumerate(samples):
        if sample is None:
            diff = max_size
            #set it to zeros
            collated_batch[i] = torch.zeros([max_size] + sample_shape)
        else:
            diff = len(sample) - max_size
            if diff == 0:
                collated_batch[i] = sample
            else:
                collated_batch[i] = torch.cat(
                    [sample, sample.new_full([-diff] + sample_shape, pad_val)]
                )
    if lenSampleShape == 1:
        collated_batch = collated_batch.unsqueeze(1)  # targets
    elif lenSampleShape == 2:
        pass  # collated_batch: [B, T, 1]
    elif lenSampleShape == 4:
        pass  # collated_batch: [B, T, C, H, W]
    return collated_batch, lengths


def collate_pad(batch):
    batch_out = {} 
    for data_type in batch[0].keys():
        pad_val = -1 if data_type == "target" else 0.0
        outputs = pad(
            [s[data_type] for s in batch if s[data_type] is not None], pad_val
        )
        #if type of ouptus is list
        if isinstance(outputs, tuple):
            c_batch, sample_lengths = outputs
            batch_out[data_type + "s"] = c_batch
            batch_out[data_type + "_lengths"] = torch.tensor(sample_lengths)
        else:
            #else it is a dictionary and we need to unpack it
            c_batch = {}
            sample_lengths = {}
            for key in outputs.keys():
                c_batch[key], sample_lengths[key] = outputs[key]
                sample_lengths[key] = torch.tensor(sample_lengths[key])
            batch_out[data_type + "s"] = c_batch
            batch_out[data_type + "_lengths"] = sample_lengths
        
    return batch_out


class DataModule(LightningDataModule):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.cfg.gpus = torch.cuda.device_count()
        self.total_gpus = self.cfg.gpus * self.cfg.trainer.num_nodes
        self.modality = self.cfg.data.modality

    def _dataloader(self, ds, sampler, collate_fn):
        return torch.utils.data.DataLoader(
            ds,
            num_workers=12,
            pin_memory=True,
            batch_sampler=sampler,
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        ds_args = self.cfg.data.dataset
        train_ds = AVDataset(
            root_dir=ds_args.root_dir,
            label_path=os.path.join(
                ds_args.root_dir, ds_args.label_dir, ds_args.train_file
            ),
            subset="train",
            modality=self.cfg.data.modality,
            audio_transform=AudioTransform("train"),
            video_transform=VideoTransform("train"),
        )
        sampler = ByFrameCountSampler(train_ds, self.cfg.data.max_frames,modality = self.modality)
        if self.total_gpus > 1:
            sampler = DistributedSamplerWrapper(sampler)
        else:
            sampler = RandomSamplerWrapper(sampler)
        return self._dataloader(train_ds, sampler, collate_pad)

    def val_dataloader(self):
        ds_args = self.cfg.data.dataset
        val_ds = AVDataset(
            root_dir=ds_args.root_dir,
            label_path=os.path.join(ds_args.root_dir, ds_args.label_dir, ds_args.val_file),
            subset="val",
            modality=self.cfg.data.modality,
            audio_transform=AudioTransform("val"),
            video_transform=VideoTransform("val"),
        )
        sampler = ByFrameCountSampler(
            val_ds, self.cfg.data.max_frames_val, shuffle=False,modality = self.modality
        )
        if self.total_gpus > 1:
            sampler = DistributedSamplerWrapper(sampler, shuffle=False, drop_last=True)
        return self._dataloader(val_ds, sampler, collate_pad)

    def test_dataloader(self):
        ds_args = self.cfg.data.dataset
        dataset = AVDataset(
            root_dir=ds_args.root_dir,
            label_path=os.path.join(ds_args.root_dir, ds_args.label_dir, ds_args.test_file),
            subset="test",
            modality=self.cfg.data.modality,
            audio_transform=AudioTransform(
                "test", snr_target=self.cfg.decode.snr_target
            ),
            video_transform=VideoTransform("test"),
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)
        return dataloader
