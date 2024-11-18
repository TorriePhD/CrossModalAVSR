from pathlib import Path
import cv2 as cv
import json
from tqdm import tqdm
basePath = Path("/home/st392/groups/grp_nlp/nobackup/autodelete/datasets/AVSpeech/datasets/")
csvPath = Path("/home/st392/groups/grp_lip/nobackup/autodelete/datasets/links/labels/AVspeech_train_transcript_lengths_seg24s_no_overlap.csv")
data = csvPath.read_text().split("\n")
# dataset,video,length,transcript
# AVspeech,dataTrain/-A9gdf3j2xo/-A9gdf3j2xo_295.165000_298.165000_video_cropped.mp4,91,4793 5034 2815 1941 4978 768 4639 4635 3003 4793
jsonPath = Path("/home/st392/code/MultiTaskAVSR/preparation/fpss.json")
jsonData = json.loads(jsonPath.read_text())
#the json is a dict with the video name as key and the fps as value
# print(jsonData["dataTrain/66ksbu_BYn0/66ksbu_BYn0_150.717233_157.223733_video.mp4"])

targetFPS = 25
newCSVData = []
failedCount = 0
totalFrames = 0 
for line in tqdm(data):
    if line == "":
        continue
    line = line.split(",")
    videoName = line[1]
    videoPath = basePath / line[0] / videoName
    if not videoPath.exists():
        print(f"Video {videoPath} does not exist")
        continue
    if ".mp3" in videoName:
        video = cv.VideoCapture(str(videoPath))
        frameCount = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        line[2] = str(frameCount)
        newCSVData.append(line)
        continue
    
    videoNameWithoutCropped = videoName.replace("_video_cropped.mp4","_video.mp4")
    fps = jsonData[videoNameWithoutCropped]
    if fps == 0:
        print(f"Video {videoPath} has an fps of 0")
        failedCount += 1
        continue
    if fps>57:
        totalFrames += int(line[2])
        newCSVData.append(line)
        continue
print(f"total Length: {totalFrames/25/60/60}")
newCSVPAth = csvPath.parent / "AVspeech_train_transcript_lengths_seg24s_only_60FPScsv"
newCSVPAth.unlink(missing_ok=True)
newCSVPAth.write_text("\n".join([",".join(line) for line in newCSVData]))


    

