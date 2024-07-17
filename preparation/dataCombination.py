from pathlib import Path

trainDatasetPath = Path("/home/st392/nobackup/autodelete/datasets/LM/crossModal/sharedAVSRSimpleFull0.001/trainval")
pretrainDatasetPath = Path("/home/st392/nobackup/autodelete/datasets/LM/crossModal/sharedAVSRSimpleFull0.001/pretrain")
testDatasetPath = Path("/home/st392/nobackup/autodelete/datasets/LM/crossModal/sharedAVSRSimpleFull0.001/test")

trainDatasetTranscriptsPaths = list(trainDatasetPath.rglob("*transcript.txt"))+list(pretrainDatasetPath.rglob("*transcript.txt"))
testDatasetTranscriptsPaths = list(testDatasetPath.rglob("*transcript.txt"))
groundTruthParentPath = Path("/home/st392/groups/grp_lip/nobackup/autodelete/datasets/fslgroup/grp_lip/compute/datasets/LRS3-TED/preprocessedRetinaface/lrs3/lrs3_text_seg24s/")
trainGroundTruthPath = [groundTruthParentPath/i.parent.parent.stem/i.parent.stem/(i.name.replace("_transcript", "")) for i in trainDatasetTranscriptsPaths]
testGroundTruthPath = [groundTruthParentPath/i.parent.parent.stem/i.parent.stem/(i.name.replace("_transcript", "")) for i in testDatasetTranscriptsPaths]
print("files loaded")
i = 0
maxLen = len(trainDatasetTranscriptsPaths)
while True:
    transcriptPath, groundTruthPath= trainDatasetTranscriptsPaths[i], trainGroundTruthPath[i]
    if not transcriptPath.exists() or not groundTruthPath.exists():
        #remove from lists
        trainDatasetTranscriptsPaths.remove(transcriptPath)
        trainGroundTruthPath.remove(groundTruthPath)
        maxLen -= 1
    else:
        i += 1
    if i >= maxLen:
        break
i = 0
maxLen = len(testDatasetTranscriptsPaths)
while True:
    transcriptPath, groundTruthPath = testDatasetTranscriptsPaths[i], testGroundTruthPath[i], 
    if not transcriptPath.exists() or not groundTruthPath.exists():
        #remove from lists
        testDatasetTranscriptsPaths.remove(transcriptPath)
        testGroundTruthPath.remove(groundTruthPath)
        maxLen -= 1
    else:
        i += 1
    if i >= maxLen:
        break
print("found files")
print(len(trainDatasetTranscriptsPaths))
print(len(testDatasetTranscriptsPaths))

trainDatasetTranscriptsText = [i.read_text() for i in trainDatasetTranscriptsPaths]
testDatasetTranscriptsText = [i.read_text() for i in testDatasetTranscriptsPaths]
trainGroundTruthText = [i.read_text() for i in trainGroundTruthPath]
testGroundTruthText = [i.read_text() for i in testGroundTruthPath]
print("files read")
# remove empty strings
i = 0
maxLen = len(trainDatasetTranscriptsText)
while True:
    if trainDatasetTranscriptsText[i] == "" or trainGroundTruthText[i] == "":
        trainDatasetTranscriptsText.remove(trainDatasetTranscriptsText[i])
        trainGroundTruthText.remove(trainGroundTruthText[i])
        maxLen -= 1
    else:
        i += 1
    if i >= maxLen:
        break
i = 0
maxLen = len(testDatasetTranscriptsText)
while True:
    if testDatasetTranscriptsText[i] == "" or testGroundTruthText[i] == "":
        testDatasetTranscriptsText.remove(testDatasetTranscriptsText[i])
        testGroundTruthText.remove(testGroundTruthText[i])
        maxLen -= 1
    else:
        i += 1
    if i >= maxLen:
        break
print("empty strings removed")
testDict = []
for i in range(len(testDatasetTranscriptsText)):
    testDict.append({"transcript": testDatasetTranscriptsText[i], "groundTruth": testGroundTruthText[i]})
trainDict = []
for i in range(len(trainDatasetTranscriptsText)):
    trainDict.append({"transcript": trainDatasetTranscriptsText[i], "groundTruth": trainGroundTruthText[i]})
import json
testPath = trainDatasetPath.parent/"test.json"
trainPath = trainDatasetPath.parent/"train.json"
with open(testPath, "w") as f:
    json.dump(testDict, f)
with open(trainPath, "w") as f:
    json.dump(trainDict, f)
print("done")





