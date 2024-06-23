import argparse

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

argparser = argparse.ArgumentParser()
argparser.add_argument("--model", type=str, required=True)
argparser.add_argument("--output_dir", type=str, required=True)
args = argparser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


train_data = load_from_disk("train_data")
test_data = load_from_disk("test_data")
tokenized_train_data = train_data.map(preprocess_function, batched=True)
tokenized_test_data = test_data.map(preprocess_function, batched=True)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

id2label = {0: "safe", 1: "jailbreak"}
label2id = {"safe": 0, "jailbreak": 1}


model = AutoModelForSequenceClassification.from_pretrained(
    args.model, num_labels=2, id2label=id2label, label2id=label2id
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return {"accuracy": (predictions == labels).mean()}


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels


training_args = TrainingArguments(
    output_dir=args.output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    eval_accumulation_steps=16,
    eval_steps=500,
    num_train_epochs=1,
    weight_decay=0.01,
    evaluation_strategy="steps",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_test_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

trainer.train()
