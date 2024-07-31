import argparse
import math
import os
import pickle
import shutil
import warnings

import ffmpeg
from data.data_module import AVSRDataLoader
from tqdm import tqdm
from utils import save_vid_aud
from pathlib import Path
from time import sleep
import fcntl

warnings.filterwarnings("ignore")

# Argument parsing
parser = argparse.ArgumentParser(description="AVspeech Preprocessing")
parser.add_argument(
    "--aud-dir",
    type=str,
    required=True,
    help="Directory where the audio sequence is stored",
)
parser.add_argument(
    "--label-dir",
    type=str,
    default="",
    help="Directory where lid.csv is saved",
)
parser.add_argument(
    "--root-dir",
    type=str,
    required=True,
    help="Root directory of preprocessed dataset",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="vox2",
    help="Name of dataset",
)
parser.add_argument(
    "--seg-duration",
    type=int,
    default=24,
    help="Max duration (second) for each segment, (Default: 24)",
)
parser.add_argument(
    "--groups",
    type=int,
    default=1,
    help="Number of threads to be used in parallel",
)
parser.add_argument(
    "--job-index",
    type=int,
    default=0,
    help="Index to identify separate jobs (useful for parallel processing)",
)
args = parser.parse_args()

# Constants
seg_aud_len = args.seg_duration * 16000
# Load data
aud_dataloader = AVSRDataLoader(modality="audio")

# Load video and audio files
filenames = [
    os.path.join(args.vid_dir, _ + ".mp4")
    for _ in open(os.path.join(args.label_dir, "vox-en.id")).read().splitlines()
]

unit = math.ceil(len(filenames) / args.groups)
files_to_process = filenames[args.job_index * unit : (args.job_index + 1) * unit]
failed = 0 
for vid_filename in tqdm(files_to_process):
    if args.landmarks_dir:
        landmarks_filename = (
            vid_filename.replace(args.vid_dir, args.landmarks_dir)[:-4] + ".pkl"
        )
        #remove /mp4/ from the path
        landmarks_filename = landmarks_filename.replace("/mp4/", "/")
        landmarks = pickle.load(open(landmarks_filename, "rb"))
    else:
        landmarks = None
    try:
        video_data = vid_dataloader.load_data(vid_filename, landmarks)
        audio_data = aud_dataloader.load_data(vid_filename)
        #catch system error too
    except (UnboundLocalError, TypeError, OverflowError, AssertionError, SystemError, RuntimeError):
        print(f"Error in loading {vid_filename} file")
        failed += 1
        continue
    if video_data is None:
        failed += 1
        continue

    # Process segments
    for i, start_idx in enumerate(range(0, len(video_data), seg_vid_len)):
        dst_vid_filename = (
            f"{vid_filename.replace(args.vid_dir, dst_vid_dir)[:-4]}_{i:02d}.mp4"
        )
        dst_aud_filename = dst_vid_filename.replace(".mp4", ".wav")
        trim_video_data = video_data[start_idx : start_idx + seg_vid_len]
        trim_audio_data = audio_data[
            :, start_idx * 640 : (start_idx + seg_vid_len) * 640
        ]
        if trim_video_data is None or trim_audio_data is None:
            continue
        video_length = len(trim_video_data)
        audio_length = trim_audio_data.size(1)
        if (
            audio_length / video_length < 560.0
            or audio_length / video_length > 720.0
            or video_length < 12
        ):
            continue

        # Save video and audio
        save_vid_aud(
            dst_vid_filename,
            dst_aud_filename,
            trim_video_data,
            trim_audio_data,
            video_fps=25,
            audio_sample_rate=16000,
        )

        # Merge video and audio
        if args.combine_av:
            in1 = ffmpeg.input(dst_vid_filename)
            in2 = ffmpeg.input(dst_aud_filename)
            out = ffmpeg.output(
                in1["v"],
                in2["a"],
                dst_vid_filename[:-4] + ".m.mp4",
                vcodec="copy",
                acodec="aac",
                strict="experimental",
                loglevel="panic",
            )
            out.run()
            os.remove(dst_aud_filename)
            os.remove(dst_vid_filename)
            shutil.move(dst_vid_filename[:-4] + ".m.mp4", dst_vid_filename)
        else:
            command = f"tar -rf {tarFile} {dst_vid_filename} {dst_aud_filename}"
            os.system(command)
            Path(dst_vid_filename).unlink()
            Path(dst_aud_filename).unlink()
            
print(f"Failed: {failed} out of {len(files_to_process)}")  