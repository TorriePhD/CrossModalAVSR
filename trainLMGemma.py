from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from pathlib import Path
import json
import torch
import torchaudio

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
print("Model and tokenizer initialized.")

# Load datasets
trainDatasetPath = Path("/home/st392/nobackup/autodelete/datasets/LM/crossModal/sharedAVSRSimpleFull0.001/train.json")
testDatasetPath = Path("/home/st392/nobackup/autodelete/datasets/LM/crossModal/sharedAVSRSimpleFull0.001/test.json")
trainDataset = json.load(trainDatasetPath.open())
testDataset = json.load(testDatasetPath.open())

# Prepare dataset inputs
def prepare_inputs(data, tokenizer, prompt):
    inputs = [prompt + " " + i["transcript"].lower() for i in data]
    labels = [i["groundTruth"].lower() for i in data]
    full_inputs = [inp + " " + lab for inp, lab in zip(inputs, labels)]
    encodings = tokenizer(full_inputs, padding=True, truncation=True, return_tensors="pt")
    labels = encodings.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return encodings, labels

prompt = "The following is a transcript of a TED talk transcribed based on visual information only. Please correct this transcript:"
trainEncodings, trainLabels = prepare_inputs(trainDataset, tokenizer, prompt)
testEncodings, testLabels = prepare_inputs(testDataset, tokenizer, prompt)

# Custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        logits = predictions[0]
        decoded_preds = tokenizer.batch_decode(logits.argmax(-1).tolist(), skip_special_tokens=True)
    else:
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
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
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-4)
args = parser.parse_args()
lr = args.lr

# Define training arguments
training_args = TrainingArguments(
    output_dir=f"/home/st392/groups/grp_lip/nobackup/archive/results/LM/LRS3/gemma{lr}",
    num_train_epochs=100,
    per_device_train_batch_size=7,
    per_device_eval_batch_size=1,
    evaluation_strategy="epoch",
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=f"/home/st392/groups/grp_lip/nobackup/archive/results/LM/logs/LRS3/gemma{lr}",
    logging_steps=677,
    save_strategy="epoch",
    learning_rate=lr,
    save_safetensors=False,
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
