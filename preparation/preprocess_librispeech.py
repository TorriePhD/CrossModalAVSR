
from tqdm import tqdm
from pathlib import Path
import torchaudio
import re
from transforms import TextTransform
import torch
parentPath = Path("/home/st392/groups/grp_avsr/nobackup/autodelete/librispeech/LibriSpeech/")
txtPaths = list((parentPath/"train-clean-100").rglob("*.txt"))+list((parentPath/"train-clean-360").rglob("*.txt"))+list((parentPath/"train-other-500").rglob("*.txt"))+list((parentPath/"dev-clean").rglob("*.txt"))+list((parentPath/"dev-other").rglob("*.txt"))+list((parentPath/"test-clean").rglob("*.txt"))+list((parentPath/"test-other").rglob("*.txt"))
data = []
for txtPath in tqdm(txtPaths):
    with open(txtPath, 'r') as file:
        for line in file:
            parts = line.strip().split(None, 1)  # Split the line into a maximum of 2 parts
            if len(parts) == 2:
                filename, transcript = parts
                data.append({
                    'filePath':txtPath,
                    'filename': filename,
                    'transcript': transcript
                })
            else:
                print(f"Skipping malformed line: {line}")
    
fps = 25
csvData = []
chars_to_ignore_regex = r"[\,\?\.\!\-\;\:\"]"
text_transform = TextTransform()
desiredSampleRate = 16000
totalLength = 0
loop = tqdm("Processing data", total=len(data))
for item in data:
    flacPath = item["filePath"].parent/f"{item['filename']}.flac"
    assert flacPath.exists(), f"FLAC file not found: {flacPath}"
    audio, sample_rate = torchaudio.load(flacPath)
    assert sample_rate == desiredSampleRate, f"Invalid sample rate: {sample_rate}"
    lengthInSec = audio.shape[1]/sample_rate
    if lengthInSec > 20:
        print(f"Skipping file {flacPath} with duration {lengthInSec} seconds")
        continue
    totalLength += lengthInSec
    lengthInFrames = lengthInSec*fps
    transcript = item['transcript']
    transcript = (
            re.sub(chars_to_ignore_regex, "", transcript)
            .upper()
            .replace("’", "'")
        )
    transcript = " ".join(transcript.split())
    token_id_str = " ".join(map(str, [_.item() for _ in text_transform.tokenize(transcript)]))
    relativePath = flacPath.relative_to(parentPath.parent)
    csvData.append(f"librispeech,{relativePath},{int(lengthInFrames)},{token_id_str}")
    loopMessage = f"Total duration: {totalLength/3600:.2f} hours"
    loop.set_postfix_str(loopMessage)
    loop.update(1)
loop.close()
csvFilePath = parentPath.parent / "librispeech_transcript_lengths.csv"
with open(csvFilePath, 'w') as file:
    file.write("\n".join(csvData))