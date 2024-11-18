import argparse
import math
import pickle
import shutil
import warnings
import torchaudio
import ffmpeg
from data.data_module import AVSRDataLoader
from tqdm import tqdm
from utils import save_vid_aud
from pathlib import Path
from time import sleep
import fcntl
import torchvision


parser = argparse.ArgumentParser(description="AVspeech Preprocessing")
parser.add_argument(
    "--job-index",
    type=int,
    default=0,
    help="Index to identify separate jobs (useful for parallel processing)",
)
parser.add_argument(
    "--groups",
    type=int,
    default=1,
    help="Number of threads to be used in parallel",
)

args = parser.parse_args()
job_index = args.job_index
groups = args.groups
avSpeechPath = Path("/home/st392/groups/grp_nlp/nobackup/autodelete/datasets/AVSpeech/datasets/AVspeech")
trainSet = avSpeechPath/"dataTrain"
trainCsv = avSpeechPath/"avspeech_train.csv"

vid_dataloader = AVSRDataLoader(modality="video",detector="retinaface",convert_gray=False)

# csv line: H1ulMfj5wRY,112.320000,116.940000,0.112500,0.345833
#/home/st392/groups/grp_nlp/nobackup/autodelete/datasets/AVSpeech/datasets/AVspeech/dataTest/H1ulMfj5wRY/H1ulMfj5wRY_112.320000_116.940000_video.mp4
frames_per_second = 25
def processSet(setPath,csvPath):
    with open(csvPath,"r") as f:
        lines = f.readlines()
    #take only the job_index-th part of the lines if the lines is divided into groups
    if groups > 1:
        unit = math.ceil(len(lines) / groups)
        lines = lines[job_index * unit : (job_index + 1) * unit]
    for line in tqdm(lines):
        line = line.strip().split(",")
        videoId, start, end, x, y = line
        
        x,y = float(x),float(y)
        videoPath = setPath/f"{videoId}/{videoId}_{start}_{end}_video.mp4"
        savePath = videoPath.parent/(videoPath.stem+"_cropped.mp4")
        if savePath.exists():
            continue
        audioPath = setPath/f"{videoId}/{videoId}_{start}_{end}_audio.mp3"
        if not audioPath.exists():
            continue
        #load in audio and resample to 16kHz if not already
        try:
            audio, sampleRate = torchaudio.load(str(audioPath))
        except:
           print(f"Error in {audioPath}")
           continue
        if sampleRate != 16000:
            audio = torchaudio.transforms.Resample(sampleRate,16000)(audio)
            torchaudio.save(audioPath,audio,16000)
        if not videoPath.exists():
            continue            
        try:
            video_data = vid_dataloader.load_data(str(videoPath), landmarks=None, transform=True,selectedFace=(x,y))
        except:
            print(f"Error in {videoPath}")
            continue
        torchvision.io.write_video(str(savePath), video_data, frames_per_second)
        videoPath.unlink()

processSet(trainSet,trainCsv)