from tqdm import tqdm
from pathlib import Path
import torchvision
import re
from transforms import TextTransform
import torch
import json 
parentPath = Path("/home/st392/groups/grp_lip/nobackup/archive/datasets/WildVSR/WildVSR")
videoPath = parentPath/"videos"
jsonPath = parentPath/"labels.json"
#header client_id       path    sentence_id     sentence        sentence_domain up_votes        down_votes      age     gender  accents variant locale  segment
#{
    # "0000.mp4": "TO A MAXIMUM OF FOUR TO FIVE SENTENCES SO NEXT LET'S TALK ABOUT WHAT TO SAY NOW THIS IS THE MILLION DOLLAR QUESTION",

jsonDict = json.load(open(jsonPath))



text_transform = TextTransform()
totalLength = 0
chars_to_ignore_regex = r"[\,\?\.\!\-\;\:\"]"
loop = tqdm("Processing data", total=len(jsonDict), position=0, leave=True)
csvData = []
fps = 25
#key is vidname, value is words
for videoName, item in jsonDict.items():
    myPath = videoPath/videoName
    video = torchvision.io.read_video(str(myPath), pts_unit='sec')[0]
    lengthInFrames = video.size(0)
    lengthInSec = lengthInFrames/fps
    if lengthInSec > 20:
        print(f"Large file {myPath} with duration {lengthInSec} seconds")
    totalLength += lengthInSec
    transcript = item
    transcript = (
            re.sub(chars_to_ignore_regex, "", transcript)
            .upper()
            .replace("’", "'")
        )
    transcript = " ".join(transcript.split())
    token_id_str = " ".join(map(str, [_.item() for _ in text_transform.tokenize(transcript)]))
    relativePath = myPath.relative_to(parentPath.parent)
    csvData.append(f"wildVSR,{relativePath},{int(lengthInFrames)},{token_id_str}")
    loopMessage = f"Total duration: {totalLength/3600:.2f} hours"
    loop.set_postfix_str(loopMessage)
    loop.update(1)
loop.close()
csvFilePath = parentPath.parent / "wildVSR_transcript_lengths.csv"
with open(csvFilePath, 'w') as file:
    file.write("\n".join(csvData))