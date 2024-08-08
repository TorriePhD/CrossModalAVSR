import os

import torch
import torchaudio
import torchvision
from pathlib import Path

def cut_or_pad(data, size, dim=0):
    """
    Pads or trims the data along a dimension.
    """
    if data.size(dim) < size:
        padding = size - data.size(dim)
        data = torch.nn.functional.pad(data, (0, 0, 0, padding), "constant")
        size = data.size(dim)
    elif data.size(dim) > size:
        data = data[:size]
    assert data.size(dim) == size
    return data


def load_video(path):
    """
    rtype: torch, T x C x H x W
    """
    vid = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
    vid = vid.permute((0, 3, 1, 2))
    return vid


def load_audio(path):
    """
    rtype: torch, T x 1
    """
    waveform, sample_rate = torchaudio.load(path[:-4] + ".wav", normalize=True)
    return waveform.transpose(1, 0)


class AVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        label_path,
        subset,
        modality,
        audio_transform,
        video_transform,
        rate_ratio=640,
    ):

        self.root_dir = root_dir

        self.modality = modality
        self.rate_ratio = rate_ratio

        self.list = self.load_list(label_path)

        self.audio_transform = audio_transform
        self.video_transform = video_transform

    def load_list(self, label_path):
        paths_counts_labels = []
        for path_count_label in open(label_path).read().splitlines():
            dataset_name, rel_path, input_length, token_id = path_count_label.split(",")
            if self.modality == "audiovisual":
                # if ("trainval" in rel_path and "lrs3" in dataset_name) or "test" in rel_path:
                if  "lrs3" in dataset_name or "test" in rel_path:
                    modalities = ["audiovisual"]
                else:
                    modalities = ["audio"]
                for modality in modalities:
                    paths_counts_labels.append(
                        (
                            dataset_name,
                            rel_path,
                            int(input_length),
                            torch.tensor([int(_) for _ in token_id.split()]),
                            modality
                    )
                )
                    
                # modalities = ["audio", "video","audiovisual"]
                # #randomly choose a modality
                # paths_counts_labels.append(
                #     (
                #         dataset_name,
                #         rel_path,
                #         int(input_length),
                #         torch.tensor([int(_) for _ in token_id.split()]),
                #         modalities[torch.randint(0,3,(1,)).item()]
                #     )
                # )
            else:
                paths_counts_labels.append(
                    (
                        dataset_name,
                        rel_path,
                        int(input_length),
                        torch.tensor([int(_) for _ in token_id.split()]),

                    )
            )
        return paths_counts_labels

    def __getitem__(self, idx):
        if self.modality == "audiovisual":
            return self.getitem_audiovisual(idx)
        dataset_name, rel_path, input_length, token_id = self.list[idx]
        path = os.path.join(self.root_dir, dataset_name, rel_path)
        if self.modality == "video":
            video = load_video(path)
            video = self.video_transform(video)
            return {"input": video, "target": token_id}
        elif self.modality == "audio":
            audio = load_audio(path)
            audio = self.audio_transform(audio)
            return {"input": audio, "target": token_id}

    def getitem_audiovisual(self, idx):
        dataset_name, rel_path, input_length, token_id, modality = self.list[idx]
        path = os.path.join(self.root_dir, dataset_name, rel_path)
        video = None
        audio = None
        if "video" in modality or "visual" in modality:
            if not Path(path).exists():
                #raise error
                print("vid path not exists: ", path)                
            video = load_video(path)
            video = self.video_transform(video)
        if "audio" in modality:
            if not Path(path[:-4] + ".wav").exists():
                #raise error
                print("audio path not exists: ", path[:-4] + ".wav")
            audio = load_audio(path)
            audio = self.audio_transform(audio)
        if modality == "audio":
            video = torch.zeros((1, 1, 88, 88))

        return {"input":{"video": video, "audio": audio}, "target": token_id}
    def __len__(self):
        return len(self.list)
