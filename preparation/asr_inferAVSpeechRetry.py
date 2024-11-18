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
# AVspeech,dataTrain/HerZkp2j5lw/HerZkp2j5lw_210.919000_215.965000_video_cropped.mp4,121,4939 3056 4742 4594 4590 2435 903 713 1393 4635 4325 4575 4854 713 4590 2 393 3282 2299 4699 2176 4577 3412 4491
csvFilePath = Path("/home/st392/groups/grp_lip/nobackup/autodelete/datasets/links/labels/AVspeech_train_transcript_lengths_seg24s_no_overlap_resampled_fixed_mp3_no_corrupt.csv")
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
        dataset,path,length,transcript = line
        videoPath = Path(args.root_dir) / args.dataset / path
        videoPath = str(videoPath).replace("_video_cropped.mp4", "_audio.mp3")
        if Path(videoPath).exists():
            files_to_process.append([str(videoPath),path,length,transcript])
        # else:
            # print(f"File {videoPath} not found")
# Label filename
print(f"Processing {len(files_to_process)} files")
label_filename = os.path.join(
    args.root_dir,
    "labels",
    f"{args.dataset}_train_transcript_lengths_seg_sampledRight_{args.seg_duration}s.csv"
    if args.groups <= 1
    else f"{args.dataset}_train_transcript_lengths_seg_sampledRight_{args.seg_duration}s.{args.groups}.{args.job_index}.csv",
)

os.makedirs(os.path.dirname(label_filename), exist_ok=True)
print(f"Directory {os.path.dirname(label_filename)} created")

f = open(label_filename, "w")

# Load ASR model
model = whisper.load_model("large-v3", device="cuda")

# Transcription
for data in tqdm(files_to_process):
    filename = data[0]
    originalpath, originalLength, originalTranscript = data[1], data[2], data[3]
    try:
        with torch.no_grad():
            #verify 16000 Hz
            audio, sr = torchaudio.load(filename)
            if sr != 16000:
                resample = torchaudio.transforms.Resample(sr, 16000)
                audio = resample(audio)
                #overwrite the file
                torchaudio.save(filename, audio, 16000)

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
        if token_id_str != originalTranscript:
            print(f"Transcript mismatch: {originalpath} {originalTranscript} {token_id_str}")
        if token_id_str:
            f.write(
                "{}\n".format(
                    f"{args.dataset},{basename},{length},{token_id_str}"
                )
            )
