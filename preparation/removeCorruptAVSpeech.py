from pathlib import Path
import torchvision
import torchaudio
from tqdm import tqdm
from multiprocessing import Pool

csvPath = Path("/home/st392/groups/grp_lip/nobackup/autodelete/datasets/links/labels/AVspeech_train_transcript_lengths_seg24s_no_overlap_resampled_fixed_mp3_no_corrupt.csv")

data = csvPath.read_text().split("\n")
data = [line.split(",") for line in data]
rootPath = Path("/home/st392/groups/grp_nlp/nobackup/autodelete/datasets/AVSpeech/datasets/")

def process_line(line):
    if int(line[2]) > 25*20 or int(line[2]) < 10 or len(line[3].split(" ")) < 3:
        return None
    if line[1][-4:] == ".mp4":
        try:
            path = rootPath / line[0] / line[1]
            audioPath = path.parent /path.name.replace("_video_cropped.mp4","_audio.mp3")
            # audioData,sample_rate = torchaudio.load(str(audioPath))

            # if audioData.shape[0] ==0 or len(audioData.shape) != 2:
            #     return None
            if path.exists() and not audioPath.exists():
                print(f"Failed to load {audioPath}")
                return None
            # videoData = torchvision.io.read_video(str(path), pts_unit="sec", output_format="THWC")[0]
            # videoLength = videoData.shape[0]/25
            # audioLength = audioData.shape[1]/sample_rate
            # lengths = [videoLength,audioLength,int(line[2])/25]
            # if max(lengths) - min(lengths) >10:
            #     print(videoLength,audioLength,int(line[2])/25)
            #     print(max(lengths) - min(lengths))
            #     return None
            # if videoData.shape[0] ==0 or len(videoData.shape) != 4:
            #     print(f"Failed to load {path} with shape {videoData.shape}")
            #     return None
            return line
        except:
            print(f"Failed to load {path}")
            return None
    else:
        try:
            path = rootPath / line[0] / line[1]   
            audioData,sample_rate = torchaudio.load(str(path))
            if audioData.shape[0] ==0 or len(audioData.shape) != 2:
                print("Failed to load audio",path)
                return None
        except:
            print(f"Failed to load audio 2 {path}")
            return None
        return line

if __name__ == '__main__':
    # results = []
    # for line in tqdm(data):
    #     results.append(process_line(line))
    with Pool() as pool:
        results = list(tqdm(pool.imap_unordered(process_line, data), total=len(data)))

    newData = [line for line in results if line is not None]
    failedCount = len(data) - len(newData)
    totalTime = sum([int(line[2]) for line in newData])
    print(f"Total time: {totalTime/3600/25} hours")
    print(f"Failed to load {failedCount} videos")
    newCSVPath = csvPath.parent / "AVspeech_train_transcript_lengths_seg24s_no_overlap_resampled_fixed_mp3_no_corrupt_no_video_only.csv"
    newCSVPath.unlink(missing_ok=True)
    newCSVPath.write_text("\n".join([",".join(line) for line in newData]))
    print("done")
