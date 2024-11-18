import argparse
import glob
import math
import os
import re

import torch
import torchvision
import torchaudio
import whisper
from tqdm import tqdm
from transforms import TextTransform
from pathlib import Path
parser = argparse.ArgumentParser(description="Transcribe into text from media")
parser.add_argument(
    "--root-dir",
    type=str,
    default="/home/st392/groups/grp_nlp/nobackup/autodelete/datasets/AVSpeech/datasets/",
    help="Root directory of preprocessed dataset",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="AVspeech",
    help="Name of dataset",
)
parser.add_argument(
    "--seg-duration",
    type=int,
    default=24,
    help="Max duration (second) for each segment, (Default: 24)",
)
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

# Constants
chars_to_ignore_regex = r"[\,\?\.\!\-\;\:\"]"
text_transform = TextTransform()
# Load video files
csvFilePath = Path(args.root_dir) /args.dataset/ f"avspeech_train.csv"
files_to_process = []
# csv line: H1ulMfj5wRY,112.320000,116.940000,0.112500,0.345833
#/home/st392/groups/grp_nlp/nobackup/autodelete/datasets/AVSpeech/datasets/AVspeech/dataTest/H1ulMfj5wRY/H1ulMfj5wRY_112.320000_116.940000_video.mp4

with open(csvFilePath, "r") as f:
    lines = f.readlines()
    # take only the job_index-th part of the lines if the lines is divided into groups
    if args.groups > 1:
        unit = math.ceil(len(lines) / args.groups)
        lines = lines[args.job_index * unit : (args.job_index + 1) * unit]
    for line in tqdm(lines):
        line = line.strip().split(",")
        videoId, start, end, x, y = line
        x, y = float(x), float(y)
        videoPath = Path(args.root_dir) / args.dataset / f"dataTrain/{videoId}/{videoId}_{start}_{end}_audio.mp3"
        if videoPath.exists():
            files_to_process.append(str(videoPath))
        # else:
            # print(f"File {videoPath} not found")
# Label filename
print(f"Processing {len(files_to_process)} files")
label_filename = os.path.join(
    args.root_dir,
    "labels",
    f"{args.dataset}_train_transcript_lengths_seg{args.seg_duration}s.csv"
    if args.groups <= 1
    else f"{args.dataset}_train_transcript_lengths_seg{args.seg_duration}s.{args.groups}.{args.job_index}.csv",
)

os.makedirs(os.path.dirname(label_filename), exist_ok=True)
print(f"Directory {os.path.dirname(label_filename)} created")

f = open(label_filename, "w")

# Load ASR model
model = whisper.load_model("large-v3", device="cuda")

# Transcription
for filename in tqdm(files_to_process):
    # Prepare destination filename
    try:
        with torch.no_grad():
            result = model.transcribe(filename)
            transcript = (
                re.sub(chars_to_ignore_regex, "", result["text"])
                .upper()
                .replace("’", "'")
            )
            transcript = " ".join(transcript.split())
    except RuntimeError:
        continue

    # Write transcript to a text file
    if transcript:
        #IcL3dkz0RYc_35.333333_41.233333_audio.mp3
        #IcL3dkz0RYc_35.333333_41.233333_video_cropped.mp4 = basname
        vidPath = filename.replace("_audio", "_video_cropped").replace(".mp3", ".mp4")
        if not os.path.exists(vidPath):
            audio = torchaudio.load(filename)
            trim_audio_data = audio[0][0]
            # get the length of the audio in frames with 25 fps
            length = trim_audio_data.size(0) // 16000 * 25
            #get filename relative to the root directory/dataset
            basename = str(Path(filename).relative_to(Path(args.root_dir) / args.dataset))
        else:
            trim_vid_data = torchvision.io.read_video(vidPath)[0]
            length = trim_vid_data.size(0)
            basename = str(Path(vidPath).relative_to(Path(args.root_dir) / args.dataset))
        token_id_str = " ".join(
            map(str, [_.item() for _ in text_transform.tokenize(transcript)])
        )
        if token_id_str:
            f.write(
                "{}\n".format(
                    f"{args.dataset},{basename},{length},{token_id_str}"
                )
            )
