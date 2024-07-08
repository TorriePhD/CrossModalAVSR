from pathlib import Path
from demo import main
import hydra
from tqdm import tqdm
@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def transcribe(cfg):
    paths = list(Path(cfg.file_path).rglob('*.mp4'))
    cfg.pretrained_model_path = "/home/st392/fsl_groups/grp_lip/compute/datasets/LRS3-TED/vsr_trlrs3_base.pth"
    for mp4Vid in tqdm(paths):
        cfg.file_path = str(mp4Vid)
        transcript = main(cfg)
        savePath = mp4Vid.parent/Path(mp4Vid.stem + "_transcript.txt")
        savePath.parent.mkdir(parents=True, exist_ok=True)
        with open(savePath, 'w') as f:
            f.write(transcript)
    
if __name__ == "__main__":
    transcribe()
    