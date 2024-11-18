from pathlib import Path 

csv_path = Path('/home/st392/groups/grp_nlp/nobackup/autodelete/datasets/AVSpeech/datasets/labels/AVspeech_train_transcript_lengths_seg_sampledRight_24s.csv')
output_path = Path('/home/st392/groups/grp_nlp/nobackup/autodelete/datasets/AVSpeech/datasets/labels/AVspeech_train_transcript_lengths_seg_sampledRight_24s_cleaned.csv')

# remove lines with "501 1 501 1"
with open(csv_path, 'r') as f:
    lines = f.readlines()
    lines = [line for line in lines if '501 1 501 1' not in line]

with open(output_path, 'w') as f:
    f.writelines(lines)