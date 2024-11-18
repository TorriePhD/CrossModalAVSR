from tqdm import tqdm
from pathlib import Path
import torchaudio
import re
from transforms import TextTransform
import torch
parentPath = Path("/home/st392/groups/grp_lip/nobackup/autodelete/datasets/links/commonVoice/")
clipPaths = [parentPath/"part1/cv-corpus-18.0-2024-06-14/en/clips", parentPath/"part2/cv-corpus-18.0-2024-06-14/en/clips"]
tsvPath = parentPath/"part1/cv-corpus-18.0-2024-06-14/en/validated.tsv"
#header client_id       path    sentence_id     sentence        sentence_domain up_votes        down_votes      age     gender  accents variant locale  segment
lines = tsvPath.read_text().split("\n")[1:]
fps = 25
text_transform = TextTransform()
desiredSampleRate = 16000
totalLength = 0
chars_to_ignore_regex = r"[\,\?\.\!\-\;\:\"]"
loop = tqdm("Processing data", total=len(lines))
csvData = []
for line in lines:
    parts = line.split("\t")
    if len(parts) >= 13:
        client_id, path, sentence_id, sentence, sentence_domain, up_votes, down_votes, age = parts[:8]
        gender, accents, variant, locale, segment = parts[8:]
        filePath = clipPaths[0]/path
        if not filePath.exists():
            filePath = clipPaths[1]/path
        item = {
            'filePath': filePath,
            'filename': path,
            'transcript': sentence
        }    
    flacPath = item['filePath']
    audio, sample_rate = torchaudio.load(flacPath)
    if sample_rate != desiredSampleRate:
        audio = torchaudio.transforms.Resample(sample_rate, desiredSampleRate)(audio)
        torchaudio.save(flacPath, audio, desiredSampleRate)
        sample_rate = desiredSampleRate
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
    csvData.append(f"commonVoice,{relativePath},{int(lengthInFrames)},{token_id_str}")
    loopMessage = f"Total duration: {totalLength/3600:.2f} hours"
    loop.set_postfix_str(loopMessage)
    loop.update(1)
loop.close()
csvFilePath = parentPath.parent / "commonVoice_transcript_lengths.csv"
with open(csvFilePath, 'w') as file:
    file.write("\n".join(csvData))