import argparse
import base64
import io
import os
import pickle

import requests
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"USING DEVICE: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained("xTRam1/safe-guard-classifier")
model = AutoModelForSequenceClassification.from_pretrained(
    "xTRam1/safe-guard-classifier"
)

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    truncation=True,
    max_length=512,
    device=torch.device(DEVICE),
)

app = FastAPI()


@app.post("/generate")
async def generate(request: dict):
    input = request["text"]
    print("INPUT:", input)
    result = classifier(input)
    print("RESULT:", result)
    return JSONResponse(content={"text": input, "result": result})


if __name__ == "__main__":
    # print("here")
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    port = args.port
    uvicorn.run(app, host="127.0.0.1", port=port)
