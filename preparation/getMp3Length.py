from pathlib import Path
import torchaudio
import json
from tqdm import tqdm
basePath = Path("/home/st392/groups/grp_lip/nobackup/autodelete/datasets/links/")
csvPath = Path("/home/st392/groups/grp_lip/nobackup/autodelete/datasets/links/labels/allAVSRNewNoMp3.csv")
data = csvPath.read_text().split("\n")
# dataset,video,length,transcript
# AVspeech,dataTrain/-A9gdf3j2xo/-A9gdf3j2xo_295.165000_298.165000_video_cropped.mp4,91,4793 5034 2815 1941 4978 768 4639 4635 3003 4793

targetFPS = 25
newCSVData = []
failedCount = 0 
for line in tqdm(data):
    if line == "":
        continue
    line = line.split(",")
    audioName = line[1]
    audioPath = basePath / line[0] / audioName
    if ".mp4" in audioName:
        audioName = audioName.replace(".mp4",".wav")
        audioPath = basePath / line[0] / audioName
    if audioPath.exists() == False:
        # print("Failed to find audio: ",audioPath)
        # failedCount += 1
        continue
    if ".mp3" in audioName or ".wav" in audioName:
        audio,sr = torchaudio.load(audioPath)
        print(audio.shape)
        frameCount = int(audio.size(1)/sr*targetFPS)
        line[2] = str(frameCount)
    newCSVData.append(line)
    
# newCSVPAth = csvPath.parent / "AVspeech_train_transcript_lengths_seg24s_no_overlap_resampled_fix_mp3.csv"
# newCSVPAth.unlink(missing_ok=True)
# newCSVPAth.write_text("\n".join([",".join(line) for line in newCSVData]))


    

