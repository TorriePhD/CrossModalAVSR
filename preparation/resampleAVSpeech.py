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
previousOutputFile = Path("/home/st392/code/MultiTaskAVSR/scripts/outputsPreprocessingAVSpeechResample/66655987_4294967294_avSpeechPreprocess.out")
doneFiles = previousOutputFile.read_text().replace("Resampled ","").split("\n")
print(doneFiles[0])

targetFPS = 25
newCSVData = []
failedCount = 0 
for line in tqdm(data):
    if line == "":
        continue
    line = line.split(",")
    videoName = line[1]
    videoPath = basePath / line[0] / videoName
    if not videoPath.exists():
        print(f"Video {videoPath} does not exist")
        continue
    if "-A9gdf3j2xo_295.165000_298.165000_video_cropped.mp4" in videoName:
        continue
    if ".mp3" in videoName or str(videoPath) in doneFiles:
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
    if fps-targetFPS < 1:
        # print(f"Video {videoPath} already has a fps of {fps}")
        newCSVData.append(line)
        continue
    if fps < 28:
        newCSVData.append(line)
        continue
    video = cv.VideoCapture(str(videoPath))

    frames = []
    while True:
        ret,frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video.release()
    if fps > 40:
        frames = frames[::2]
    elif fps >=28 and fps <= 40:
        newFrames = []
        for i,frame in enumerate(frames):
            if i%6 != 0:
                newFrames.append(frame)
        frames = newFrames
    else:
        print(f"Video {videoPath} has an fps of {fps}")
        failedCount += 1
        continue
    videoPath.unlink()
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    if len(frames) == 0:
        print(f"Video {videoPath} has 0 frames")
        failedCount += 1
        continue
    out = cv.VideoWriter(str(videoPath),fourcc,targetFPS,(frames[0].shape[1],frames[0].shape[0]))
    for frame in frames:
        out.write(frame)
    out.release()
    line[2] = str(len(frames))
    newCSVData.append(line)
    print(f"Resampled {videoPath}")
newCSVPAth = csvPath.parent / "AVspeech_train_transcript_lengths_seg24s_no_overlap_resampled.csv"
newCSVPAth.unlink(missing_ok=True)
newCSVPAth.write_text("\n".join([",".join(line) for line in newCSVData]))


    

