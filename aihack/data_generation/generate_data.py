import asyncio
import json
import os

from datasets import Dataset, load_dataset
from langchain_openai import ChatOpenAI

from aihack.aihack.data_generation.malicious_instruction_generator import (
    JailBreakExample,
    MaliciousInstructionGenerator,
)
from aihack.aihack.data_generation.repo import JailBreakExampleRepo

DATA_FILE_NAME = "malicious_data.json"
MAX_CONCURRENT_REQUESTS = 5
MAX_EXAMPLES_TO_GENERATE = 2600


async def main():
    examples = []
    if os.path.exists(DATA_FILE_NAME):
        with open(DATA_FILE_NAME) as f:
            examples = [JailBreakExample.from_json(example) for example in json.load(f)]

    jailbreak_dataset = load_dataset("jackhhao/jailbreak-classification")

    def filter_for_type(data: Dataset, type: str) -> Dataset:
        return data.filter(lambda example: example["type"] == type)

    jailbreak_dataset_train = filter_for_type(jailbreak_dataset["train"], "jailbreak")
    jailbreak_example_repo_train = JailBreakExampleRepo(jailbreak_dataset_train)

    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.9,
    )

    malicious_data_generator = MaliciousInstructionGenerator(
        model, jailbreak_example_repo_train
    )

    while True:
        if len(examples) >= MAX_EXAMPLES_TO_GENERATE:
            print(f"Generated {len(examples)} examples. Stopping the generation")
            break

        print("=" * 50)
        print(
            f"Generating malicious data iteration. Current examples count: {len(examples)}. Target examples count: {MAX_EXAMPLES_TO_GENERATE}"
        )
        malicious_data = await malicious_data_generator.generate_malicious_instruction(
            max_conccurrent_requests=MAX_CONCURRENT_REQUESTS
        )
        examples.extend(malicious_data)
        MaliciousInstructionGenerator.save_to_file(examples, DATA_FILE_NAME)
        print(f"Generated {len(malicious_data)} malicious data examples")


if __name__ == "__main__":
    asyncio.run(main())
