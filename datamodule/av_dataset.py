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
    if path[-4:] == ".mp4":
        waveform, sample_rate = torchaudio.load(path[:-4] + ".wav", normalize=True)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
            sample_rate = 16000
            #save the resampled audio
            torchaudio.save(path[:-4] + ".wav", waveform, 16000)
    else:
        waveform, sample_rate = torchaudio.load(path, normalize=True)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
            sample_rate = 16000
            #save the resampled audio
            torchaudio.save(path, waveform, 16000)
    #if stereo, convert to mono
    if waveform.shape[0] == 2:
        waveform = waveform.mean(0, keepdim=True)


    if len(waveform.shape) == 1:
        print(f"Warning: {path} is only one channel")
        waveform = waveform.unsqueeze(0)
        

    return waveform.transpose(1, 0)

def load_audio_logMel(path):
    #change path's extension to .pt
    path = str(Path(path).with_suffix(".pt"))
    if not Path(path).exists():
        print("audio path not exists: ", path)
        return None
    audio = torch.load(path)
    return audio
    

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
        lowResource=False
    ):

        self.root_dir = root_dir

        self.modality = modality
        self.rate_ratio = rate_ratio
        self.lowResource = lowResource
        self.list = self.load_list(label_path)
        if "WildVSR" in self.list[0][0]:   
            self.modality = "video"
        self.audio_transform = audio_transform
        self.video_transform = video_transform

    def load_list(self, label_path):
        paths_counts_labels = []
        for path_count_label in open(label_path).read().splitlines():
            if len(path_count_label.split(",")) != 4:
                print("error on: ", path_count_label)
                continue
            dataset_name, rel_path, input_length, token_id = path_count_label.split(",")
            if self.modality == "audiovisual":
                if not self.lowResource:
                    if "WildVSR" in dataset_name:
                        modalities = ["video"]
                    elif  "lrs3" in dataset_name or ".mp4" in rel_path:
                        modalities = ["audiovisual"]
                    else:
                        modalities = ["audio"]
                else:
                    if "lrs3" in dataset_name or "test" in rel_path:
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
        return self.getitem_audiovisual(idx)


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
            audio = load_audio_logMel(path)
        if modality == "audio":
            video = torch.zeros((1, 1, 88, 88))
        elif modality == "video":
            audio = torch.zeros((1, 1))
        #make all torch.float16
        video = video.half()
        audio = audio.half()
        return {"input":{"video": video, "audio": audio}, "target": token_id}
    def __len__(self):
        return len(self.list)
