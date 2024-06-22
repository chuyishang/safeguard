import requests
import torch
import argparse
import torch
import base64
import io
import pickle
import os
import uvicorn

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from fastapi import FastAPI
from fastapi.responses import JSONResponse

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"USING DEVICE: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained("ProtectAI/deberta-v3-base-prompt-injection-v2")
model = AutoModelForSequenceClassification.from_pretrained("ProtectAI/deberta-v3-base-prompt-injection-v2")

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
async def embed(request: dict):
    input = request["text"]
    result = classifier(input)
    return JSONResponse(content={"text": input, "result": result})

if __name__ == "__main__":
    # print("here")
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    port = args.port
    uvicorn.run(app, host="127.0.0.1", port=port)
