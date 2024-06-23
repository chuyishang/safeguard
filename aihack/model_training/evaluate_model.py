import argparse
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk

def compute_metrics(predictions, labels):
    accuracy = (np.array(predictions) == np.array(labels)).mean()
    return {"accuracy": accuracy}

def preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, return_tensors="pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
args = parser.parse_args()

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(
    args.model, num_labels=2, id2label={0: "safe", 1: "jailbreak"}, label2id={"safe": 0, "jailbreak": 1}
).to(device)
tokenizer = AutoTokenizer.from_pretrained(args.model)

# Load test data
test_data = load_from_disk("test_data")

# Generate predictions and references
references = [example["label"] for example in test_data]
predictions = []
from tqdm import tqdm
for example in tqdm(test_data, total=len(test_data)):
    inputs = preprocess_function(example, tokenizer)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to GPU
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    predictions.append(prediction)

# Compute the metrics
metrics = compute_metrics(predictions, references)
print("Accuracy: ", metrics["accuracy"])

