from pathlib import Path

allAVSRPath = Path("/home/st392/groups/grp_lip/nobackup/autodelete/datasets/links/labels/allAVSRNew.csv")

allAVSRData = allAVSRPath.read_text().split("\n")
newAllAVSRPath = Path("/home/st392/groups/grp_lip/nobackup/autodelete/datasets/links/labels/allAVSRNewNoMp3.csv")
newAllAVSRData = []
for line in allAVSRData:
    if "_audio.mp3" in line:
        continue
    newAllAVSRData.append(line)
newAllAVSRPath.write_text("\n".join(newAllAVSRData))
print("done")