
import whisper
import torch
from pathlib import Path
from tqdm import tqdm
class LanguageID:
    def __init__(self):
        self.device =  "cpu"
        self.model = whisper.load_model("base", self.device)
        print("model loaded")

    def detect_language(self, audio):
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        _, probs = self.model.detect_language(mel)
        return max(probs, key=probs.get)
    
    def detect_language_from_file(self, path):
        audio = whisper.load_audio(path)
        return self.detect_language(audio)
    
    def detectAllLanguages(self, allFolders):
        print("loading all mp3 files")
        # wavFiles = parentPath.rglob("*.mp3")
        wavFiles = []
        for folder in allFolders:
            wavFiles.extend(list(folder.rglob("*.mp3")))

        print("detecting languages")
        for wavFile in tqdm(list(wavFiles)):
            try:
                language = self.detect_language_from_file(wavFile)
            except Exception as e:
                print("error", e)
                continue
            if language != "en":
                #remove the wavfile
                print(f"removing {wavFile}, detected language: {language}")
                wavFile.unlink()

if __name__ == "__main__":
    langID = LanguageID()
    parentPath = Path("/home/st392/groups/grp_nlp/nobackup/autodelete/datasets/AVSpeech/datasets/AVspeech/")
    allFolders = list((parentPath/"dataTest").iterdir())+list((parentPath/"dataTrain").iterdir())
    print("total folders", len(allFolders))
    #add argparser with index and max arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max", type=int, default=1)
    args = parser.parse_args()
    #get the index/max percentage 
    chunkSize = len(allFolders)//args.max
    allFolders = allFolders[args.index*chunkSize: (args.index+1)*chunkSize]
    print("processing", len(allFolders))
    print(f"with index {args.index} and max {args.max}")
    #get the folder
    
    

    langID.detectAllLanguages(allFolders)
    
        