from transformers import AutoTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from pathlib import Path
import json
import torch
import torchaudio


# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("grammarly/coedit-large")
model = T5ForConditionalGeneration.from_pretrained("grammarly/coedit-large")
print("Model and tokenizer initialized.")
# Load datasets
trainDatasetPath = Path("/home/st392/nobackup/autodelete/datasets/LM/crossModal/sharedAVSRSimpleFull0.001/train.json")
testDatasetPath = Path("/home/st392/nobackup/autodelete/datasets/LM/crossModal/sharedAVSRSimpleFull0.001/test.json")
trainDataset = json.load(trainDatasetPath.open())
testDataset = json.load(testDatasetPath.open())

prompt = "The following is a transcript of a TED talk transcribed based on visual information only. Please correct this transcript:"


# Prepare dataset inputs
trainDatasetInputs = [prompt + " " + i["transcript"].lower() for i in trainDataset]
testDatasetInputs = [prompt + " " + i["transcript"].lower() for i in testDataset]

# Tokenize inputs and labels
trainEncodings = tokenizer(trainDatasetInputs, padding=True, truncation=True, return_tensors="pt")
testEncodings = tokenizer(testDatasetInputs, padding=True, truncation=True, return_tensors="pt")
trainLabels = tokenizer([i["groundTruth"].lower() for i in trainDataset], padding=True, truncation=True, return_tensors="pt")
testLabels = tokenizer([i["groundTruth"].lower() for i in testDataset], padding=True, truncation=True, return_tensors="pt")

# Custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])
        return item

    def __len__(self):
        return len(self.encodings.input_ids)
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Extract logits (the first element of the predictions tuple) and convert to list of lists
    logits = predictions[0]  # Assuming logits are the first element
    decoded_preds = tokenizer.batch_decode(logits.argmax(-1).tolist(), skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)

    wer = 0
    wer_length = 0

    for pred, label in zip(decoded_preds, decoded_labels):
        wer += torchaudio.functional.edit_distance(pred.lower().split(), label.lower().split())
        wer_length += len(label.lower().split())

    return {"wer": wer / wer_length}

# Create dataset objects
trainDataset = CustomDataset(trainEncodings, trainLabels)
testDataset = CustomDataset(testEncodings, testLabels)
print("Datasets loaded.")
# Define training arguments set 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-4)
args = parser.parse_args()
lr = args.lr
Path("/home/st392/groups/grp_lip/nobackup/archive/results/LM/LRS3/").mkdir(parents=True, exist_ok=True)
Path("/home/st392/groups/grp_lip/nobackup/archive/results/LM/logs/LRS3/").mkdir(parents=True, exist_ok=True)
training_args = TrainingArguments(
    output_dir=f"/home/st392/groups/grp_lip/nobackup/archive/results/LM/LRS3/{lr}",
    num_train_epochs=100,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=20,
    evaluation_strategy="epoch",
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=f"/home/st392/groups/grp_lip/nobackup/archive/results/LM/logs/LRS3/{lr}",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=lr,
    
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=trainDataset,
    eval_dataset=testDataset,
    compute_metrics=compute_metrics
)
print("starting training...")
# Train the model
trainer.train()
