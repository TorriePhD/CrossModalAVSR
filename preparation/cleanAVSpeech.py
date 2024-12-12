from pathlib import Path 

csv_path = Path('/home/st392/groups/grp_lip/nobackup/autodelete/datasets/links/labels/AVspeech_new.csv')
output_path = Path('/home/st392/groups/grp_lip/nobackup/autodelete/datasets/links/labels/AVspeech_newNoBad.csv')

# remove lines with "501 1 501 1"
with open(csv_path, 'r') as f:
    lines = f.readlines()
    lines = [line for line in lines if '501 1 501 1' not in line]

with open(output_path, 'w') as f:
    f.writelines(lines)

#remove lines with .mp3 extension
lines = [line for line in lines if '.mp3' not in line]
NoMP3_path = Path('/home/st392/groups/grp_lip/nobackup/autodelete/datasets/links/labels/AVspeech_newNoBad_NoMP3.csv')
with open(NoMP3_path, 'w') as f:
    f.writelines(lines)