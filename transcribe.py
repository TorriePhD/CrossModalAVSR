from pathlib import Path
from demo import main
import hydra
from tqdm import tqdm
@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def transcribe(cfg):
    people = sorted(list(Path(cfg.file_path).iterdir()))
    index = cfg.index   
    cfg.crop = False
    paths = list(people[index].rglob('*.mp4'))
    # paths = sorted(list(Path(cfg.file_path).rglob('*.mp4')))
    print(f"Transcribing {len(paths)} videos.")
    dirSavePath = Path("/home/st392/nobackup/autodelete/datasets/LM") /Path(cfg.pretrained_model_path).relative_to("/home/st392/groups/grp_lip/nobackup/archive/results/").parent
    dirSavePath.mkdir(parents=True, exist_ok=True)
    print(dirSavePath)
    for mp4Vid in tqdm(paths):
        cfg.file_path = str(mp4Vid)
        savePath = dirSavePath/mp4Vid.parent.parent.stem/mp4Vid.parent.stem/(mp4Vid.stem + "_transcript.txt")
        saveErrorPath = dirSavePath/"Errors"/(mp4Vid.parent.stem + "_" + mp4Vid.stem + "_transcriptError.txt")
        if savePath.exists() or saveErrorPath.exists():
            print(f"Skipping {mp4Vid.stem} as it already exists.")
            continue
        try:
            transcript = main(cfg)
        except Exception as e:
            #if keyboard interupt then stop
            if isinstance(e, KeyboardInterrupt):
                break
            saveErrorPath.parent.mkdir(parents=True, exist_ok=True)
            with open(saveErrorPath, 'w') as f:
                f.write("Error")
            continue
        savePath.parent.mkdir(parents=True, exist_ok=True)
        with open(savePath, 'w') as f:
            f.write(transcript)
    
if __name__ == "__main__":
    transcribe()
    