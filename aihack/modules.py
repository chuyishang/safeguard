import abc
import openai
import torch
import os
import yaml
import asyncio
import aiohttp

from dotenv import load_dotenv
from utils import DotDict

# Set up environment, api_keys
load_dotenv()  # Load environment variables from a .env file
oai_key = os.getenv('OPENAI_API_KEY')
device = "cuda" if torch.cuda.is_available() else "cpu"


# Load configs. Currently configs are hardcoded, which is bad.
config_path = "aihack/configs/default.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
cfg = DotDict(config)

class BaseModel(abc.ABC):
    to_batch = False

    def __init__(self, gpu_number):
        if gpu_number is not None:
            self.dev = f'cuda:{gpu_number}' if device == 'cuda' else device
        if gpu_number is None:
            self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        Every module should have a `forward` class that takes in inputs and returns the outputs. 
        This function should implement the base functionality of the module.
        """
        pass

class GPT(BaseModel):
    name = 'gpt'
    to_batch = False
    requires_gpu = False

    def __init__(self, gpu_number=0, max_tries=1):
        super().__init__(gpu_number=gpu_number)
        # TODO: modify the prompting mechanism
        self.temperature = cfg.gpt.temperature
        self.n_votes = cfg.gpt.n_votes
        self.model = cfg.gpt.model
        self.max_tries = cfg.gpt.max_tries
        self.frequency_penalty = cfg.gpt.frequency_penalty
        self.presence_penalty = cfg.gpt.presence_penalty
        self.max_tokens = cfg.gpt.max_tokens

    @staticmethod
    def call_llm(prompt, model,
                 frequency_penalty=0, presence_penalty=0,
                 max_tokens=1000, n=1, temperature=1, max_tries=3):
        for _ in range(max_tries):
            try:
                completion = openai.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": prompt}],
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    max_tokens=max_tokens,
                    n=n,
                    temperature=temperature)

                output_message = completion.choices[0].message.content
                return output_message

            except Exception as e:
                logger.error(f"ERROR: couldn't call GPT with: {e}")
                logger.critical(f"trying again")
                continue

        logger.error("ERROR: could not get response from LLM")

        return None

    def forward(self, prompt):
        response = GPT.call_llm(prompt, self.model, self.frequency_penalty,
                                 self.presence_penalty, self.max_tokens, self.n_votes,
                                 self.temperature, self.max_tries)

        return response


class Detector(BaseModel):
    name = 'Detector'
    requires_gpu = True

    def __init__(self, gpu_number=None, port_number=8000):
        super().__init__(gpu_number)
        self.url = f"http://localhost:{port_number}/generate"
    
    @staticmethod
    async def send_request(url, data, delay=0):
        await asyncio.sleep(delay)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as resp:
                output = await resp.json()
        return output
    
    @staticmethod
    async def run(url, texts: list) -> dict:
        response = []
        # payloads = []
        for q in texts:
            payload = (
                url,
                {
                    "text": f"{q}"
                },
            )
            response.append(Detector.send_request(*payload))
        
        rets = await asyncio.gather(*response)
        outputs = []
        for ret in rets:
            outputs.append((ret["text"], ret["result"]))
            # print(ret["text"])
        response = None
        return outputs

    def forward(self, inputs):
        # print("IMAGE_LIST_TYPE", type(image_list[0]))
        """Assumes that image_list and questions are same length"""
        return asyncio.run(self.run(self.url, inputs))
