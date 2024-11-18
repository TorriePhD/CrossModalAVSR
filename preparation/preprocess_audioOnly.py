import argparse
import math
import pickle
import shutil
import warnings

import ffmpeg
from data.data_module import AVSRDataLoader
from tqdm import tqdm
from utils import save2aud
from pathlib import Path
from time import sleep
import fcntl

warnings.filterwarnings("ignore")

# Argument parsing
parser = argparse.ArgumentParser(description="VoxCeleb2 Preprocessing")
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
# dst_vid_dir = os.path.join(
    # args.root_dir, args.dataset, f"{args.dataset}_video_seg{args.seg_duration}s"
# )
aud_dataloader = AVSRDataLoader(modality="audio")
 
# Load video and audio files
filenames = list(dst_vid_dir.rglob("*.wav"))

unit = math.ceil(len(filenames) / args.groups)
files_to_process = filenames[args.job_index * unit : (args.job_index + 1) * unit]
failed = 0 
for aud_filename in tqdm(files_to_process):
    landmarks = None
    try:
        audio_data = aud_dataloader.load_data(aud_filename)
        #catch system error too
    except (UnboundLocalError, TypeError, OverflowError, AssertionError, SystemError, RuntimeError):
        print(f"Error in loading {aud_filename} file")
        failed += 1
        continue
    if video_data is None:
        failed += 1
        continue

    # Process segments
    for i, start_idx in enumerate(range(0, len(video_data), seg_vid_len)):
        dst_aud_filename = (
            f"{str(aud_filename).replace(args.vid_dir, dst_vid_dir)[:-4]}_{i:02d}.wav"
        )
        trim_audio_data = audio_data[
            :, start_idx * 640 : (start_idx + seg_vid_len) * 640
        ]
        if trim_audio_data is None:
            continue
        audio_length = trim_audio_data.size(1)

        # Save video and audio
        
        save2aud(dst_aud_filename, trim_audio_data, 16000)
            
print(f"Failed: {failed} out of {len(files_to_process)}")  