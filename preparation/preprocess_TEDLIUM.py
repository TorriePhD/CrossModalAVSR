import torchaudio
from pathlib import Path
from tqdm import tqdm
import argparse
import torch
from transforms import TextTransform
import re

def parse_stm(file_path):
    stm_data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(None, 6)  # Split the line into a maximum of 7 parts
            if len(parts) == 7:
                filename, speaker_id, channel, start_time, end_time, label, transcript = parts
                stm_data.append({
                    'filename': filename,
                    'speaker_id': speaker_id,
                    'channel': channel,
                    'start_time': float(start_time),
                    'end_time': float(end_time),
                    'label': label,
                    'transcript': transcript
                })
            else:
                print(f"Skipping malformed line: {line}")

    return stm_data
chars_to_ignore_regex = r"[\,\?\.\!\-\;\:\"]"
text_transform = TextTransform()
parentSTMPath = Path("/home/st392/groups/grp_avsr/nobackup/autodelete/TEDLIUM/TEDLIUM_release-3/legacy/")
stmPaths = list((parentSTMPath/"train/stm/").rglob("*.stm"))+list((parentSTMPath/"dev/stm/").rglob("*.stm"))+list((parentSTMPath/"test/stm/").rglob("*.stm"))
fps = 25
csvLines = []
csvFilePath = parentSTMPath.parent.parent / "TEDLIUM_train_transcript_lengths.csv"
for stm_path in tqdm(stmPaths):
    assert stm_path.exists(), f"STM file not found: {stm_path}"
    assert stm_path.suffix == ".stm", f"Invalid STM file: {stm_path}"
    assert stm_path.stat().st_size > 0, f"Empty STM file: {stm_path}"


    stm_data = parse_stm(stm_path)
    max_duration = 20
    target_sample_rate = 16000
    stm_data = [item for item in stm_data if item['end_time'] - item['start_time'] <= max_duration and item["transcript"] != "ignore_time_segment_in_scoring"]
    sphFilePath = stm_path.parent.parent / "sph"/ stm_path.with_suffix(".sph").name
    assert sphFilePath.exists(), f"SPH file not found: {sphFilePath}"
    audio, sample_rate = torchaudio.load(sphFilePath)
    if sample_rate != target_sample_rate:
        audio = torchaudio.functional.resample(
            audio, sample_rate, target_sample_rate
        )
        sample_rate = target_sample_rate
    audio = audio[0].numpy()
    audio_duration = audio.shape[0] / sample_rate
    parentSavePath = stm_path.parent.parent /"wav"
    
    parentSavePath.mkdir(exist_ok=True, parents=True)
    savedFilePaths = []
    for item in stm_data:
        itemSavePath = parentSavePath / item['filename'] /f"{item['filename']}_{item['start_time']:.2f}_{item['end_time']:.2f}.wav"
        itemSavePath.parent.mkdir(exist_ok=True, parents=True)
        start_frame = int(item['start_time'] * sample_rate)
        end_frame = int(item['end_time'] * sample_rate)
        torchaudio.save(itemSavePath, torch.tensor(audio[start_frame:end_frame]).unsqueeze(0), sample_rate)
        savedFilePaths.append(itemSavePath)

    #TEDLIUM,lrs3_video_seg24s/test/OIlSXRCSBSI/00002.wav,length,4575 2134 2887 2733 549 2535
    for path,item in zip(savedFilePaths,stm_data):
        transcript = item['transcript']
        transcript = (
                re.sub(chars_to_ignore_regex, "", transcript)
                .upper()
                .replace("’", "'")
            )
        transcript = " ".join(transcript.split())
        token_id_str = " ".join(map(str, [_.item() for _ in text_transform.tokenize(transcript)]))

        relativePath = path.relative_to(csvFilePath.parent)
        length = int((item['end_time'] - item['start_time']) *fps)
        csvLines.append(f"TEDLIUM,{relativePath},{length},{token_id_str}")
with open(csvFilePath, 'w') as file:
    file.write("\n".join(csvLines))