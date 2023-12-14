from pathlib import Path
import os

scriptPath = "/home/st392/code/MultiTaskAVSR/scripts/trascribe.sh"
paths = Path("/home/st392/fsl_groups/grp_nlp/compute/NLP/LRS3-TED/datasets")
for split in paths.iterdir():
    for speaker in split.iterdir():
        transcriptCount = len(list(speaker.glob("*transcript.txt")))
        txtCount = len(list(speaker.glob("*.txt")))
        if transcriptCount == txtCount-transcriptCount:
            print("done")
            continue
        scriptCMD = f"sbatch {scriptPath} {str(speaker)}"
        os.system(scriptCMD)