from pathlib import Path

avSpeechPath = Path("/home/st392/groups/grp_nlp/nobackup/autodelete/datasets/AVSpeech/datasets/labels/AVspeech_train_transcript_lengths_seg_sampledRight_24s_cleaned.csv")
lrs3TestPath = Path("/home/st392/groups/grp_lip/nobackup/autodelete/datasets/links/labels/lrs3_test_transcript_lengths_seg24s.csv")

lrs3Data = lrs3TestPath.read_text().split("\n")
avSpeechData = avSpeechPath.read_text().split("\n")
avSpeechData = [line.split(",") for line in avSpeechData]
lrs3Data = [line.split(",") for line in lrs3Data]
lrs3Dict = {}
for line in lrs3Data:
    youtubeId = line[1].split("/")[2]
    segment = line[1].split("/")[3].split(".")[0]
    if youtubeId not in lrs3Dict:
        lrs3Dict[youtubeId] = {}
    lrs3Dict[youtubeId][segment] = line
avSpeechDataDict = {}
for line in avSpeechData:
    youtubeId = line[1].split("/")[1]
    segment = line[1].split("/")[-1]
    if youtubeId not in avSpeechDataDict:
        avSpeechDataDict[youtubeId] = {}
    avSpeechDataDict[youtubeId][segment] = line
newAVSpeechData = []
for youtubeId in lrs3Dict:
    if youtubeId in avSpeechDataDict:
        print(f"found {youtubeId}")
        del avSpeechDataDict[youtubeId]
        assert youtubeId not in avSpeechDataDict

newCSVPath = Path("/home/st392/groups/grp_nlp/nobackup/autodelete/datasets/AVSpeech/datasets/labels/AVspeech_train_transcript_lengths_seg2_sampled_right_4s_cleaned_no_overlap.csv")
totalLength = 0
newAVSpeechData = []
for youtubeId in avSpeechDataDict:
    for segment in avSpeechDataDict[youtubeId]:
        newAVSpeechData.append(avSpeechDataDict[youtubeId][segment])
        totalLength += int(avSpeechDataDict[youtubeId][segment][2])
print(f"total length of newAVSpeechData: {totalLength//25//60//60} hours")
newCSVPath.write_text("\n".join([",".join(line) for line in newAVSpeechData]))
print("done")
