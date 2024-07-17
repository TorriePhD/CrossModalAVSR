from pathlib import Path
import os

scriptPath = "/home/st392/code/MultiTaskAVSR/scripts/trascribe.sh"
paths = Path("/home/st392/groups/grp_lip/nobackup/autodelete/datasets/fslgroup/grp_lip/compute/datasets/LRS3-TED/preprocessedRetinaface/lrs3/lrs3_video_seg24s")
modelPath = Path("/home/st392/groups/grp_lip/nobackup/archive/results/crossModal/sharedAVSRSimpleFull0.001/model_avg_10.pth")
startCounter = 0
counterend = 10
startCounter *=1000
counterend *=1000
split = paths/"test"

# for split in paths.iterdir():
#     scriptCMD = f"sbatch {scriptPath} {str(split)} {str(modelPath)} "
#     os.system(scriptCMD)
    # for speaker in split.iterdir():
        # transcriptCount = len(list(speaker.glob("*transcript.txt")))
        # txtCount = len(list(speaker.glob("*.txt")))
        # if transcriptCount == txtCount-transcriptCount:
        #     print("done")
        #     continue
length = len(list(split.iterdir()))-1
#start jobs in 1000 batches
for i in range(0, length, 1000):
    if i<startCounter:
        continue
    if i>counterend:
        break
    # scriptCMD = f"sbatch --array={i-5000}-{min(i+999, length)-5000} {scriptPath} {str(split)} {str(modelPath)} "
    scriptCMD = f"sbatch --array={i}-{min(i+999, length)} {scriptPath} {str(split)} {str(modelPath)} "
    os.system(scriptCMD)
    print(scriptCMD)



    # scriptCMD = f"sbatch --array=0-{} {scriptPath} {str(split)} {str(modelPath)} "
    # print(scriptCMD)
    # os.system(scriptCMD)
    # break