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
parser = argparse.ArgumentParser(description="VoxCeleb2 Preprocessing")
parser.add_argument(
    "--vid-dir",
    type=str,
    required=True,
    help="Directory where the video sequence is stored",
)
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
    "--landmarks-dir",
    type=str,
    default=None,
    help="Directory of landmarks",
)
parser.add_argument(
    "--detector",
    type=str,
    help="Type of face detector",
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
    "--combine-av",
    type=lambda x: (str(x).lower() == "true"),
    default=False,
    help="Merges the audio and video components to a media file",
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
def atomic_touch(file_path):
    with open(file_path, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)

def extract_archive(datasetPath, datasetDescription):
    datasetPath = Path(datasetPath)
    savePath = Path("/tmp/") / datasetDescription
    savePath.mkdir(parents=True, exist_ok=True)
    workingFile = savePath / f"{datasetPath.stem}working.txt"
    doneFile = savePath / f"{datasetPath.stem}done.txt"

    try:
        atomic_touch(workingFile)
        if not doneFile.exists():
            print("Extracting dataset...")

            if datasetPath.is_file() and datasetPath.suffix in [".zip", ".tar"]:
                if datasetPath.suffix == ".zip":
                    import zipfile
                    print(datasetPath.exists())
                    with zipfile.ZipFile(datasetPath, "r") as zip_ref:
                        print("extracting zip")
                        print(savePath)
                        zip_ref.extractall(savePath)
                elif datasetPath.suffix == ".tar":
                    import tarfile
                    with tarfile.open(datasetPath, "r") as tar_ref:
                        print("extracting tar")
                        tar_ref.extractall(savePath)

                doneFile.touch()
                print("Dataset extracted.")
            else:
                print("doesn't exist")
                raise FileNotFoundError(f"No archive found at {datasetPath}")
        else:
            print("Dataset already extracted.")
    except FileNotFoundError:
        print("Waiting for dataset extraction...")
        while not doneFile.exists():
            sleep(10)
        print("Dataset extracted.")
    finally:
        if workingFile.exists():
            try:
                workingFile.unlink()
            except:
                pass

    return savePath

zipPath = Path("/home/st392/fsl_groups/grp_lip/compute/datasets/VoxCeleb2/vox2_aac.zip")
extractedPath = extract_archive(zipPath, "vox2")
args.aud_dir = extractedPath
vidPath = Path("/home/st392/fsl_groups/grp_lip/compute/datasets/VoxCeleb2/vox2_mp4.zip")
extractedVidPath = extract_archive(vidPath, "vox2")
args.vid_dir = extractedVidPath
landmarkPath = Path("/home/st392/fsl_groups/grp_lip/compute/datasets/VoxCeleb2/single.zip")
extractedLandmarkPath = extract_archive(landmarkPath, "vox2")
args.landmarks_dir = extractedLandmarkPath
# Constants
seg_vid_len = args.seg_duration * 25
seg_aud_len = args.seg_duration * 16000
dst_vid_dir = os.path.join(
    args.root_dir, args.dataset, f"{args.dataset}_video_seg{args.seg_duration}s"
)
tarFile = Path(dst_vid_dir)/f"{args.dataset}_video_seg{args.seg_duration}s{args.job_index}.tar"
if tarFile.exists():
    print(f"Tar file {tarFile} already exists.")
    exit(0)
# Load data
vid_dataloader = AVSRDataLoader(
    modality="video", detector=args.detector, convert_gray=False
)
aud_dataloader = AVSRDataLoader(modality="audio")

# Load video and audio files
filenames = [
    os.path.join(args.vid_dir, _ + ".mp4")
    for _ in open(os.path.join(args.label_dir, "vox-en.id")).read().splitlines()
]

unit = math.ceil(len(filenames) / args.groups)
files_to_process = filenames[args.job_index * unit : (args.job_index + 1) * unit]

for vid_filename in tqdm(files_to_process):
    if args.landmarks_dir:
        landmarks_filename = (
            vid_filename.replace(args.vid_dir, args.landmarks_dir)[:-4] + ".pkl"
        )
        landmarks = pickle.load(open(landmarks_filename, "rb"))
    else:
        landmarks = None
    try:
        video_data = vid_dataloader.load_data(vid_filename, landmarks)
        aud_filename = vid_filename.replace(args.vid_dir, args.aud_dir)[:-4] + ".wav"
        audio_data = aud_dataloader.load_data(aud_filename)
    except (UnboundLocalError, TypeError, OverflowError, AssertionError):
        continue
    if video_data is None:
        continue

    # Process segments
    for i, start_idx in enumerate(range(0, len(video_data), seg_vid_len)):
        dst_vid_filename = (
            f"{vid_filename.replace(args.vid_dir, dst_vid_dir)[:-4]}_{i:02d}.mp4"
        )
        dst_aud_filename = (
            f"{aud_filename.replace(args.aud_dir, dst_vid_dir)[:-4]}_{i:02d}.wav"
        )
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
if len(list(Path(dst_vid_dir).rglob("*.mp4")))==0 and len(list(Path(dst_vid_dir).rglob("*.wav")))==0:
    shutil.rmtree(dst_vid_dir)
