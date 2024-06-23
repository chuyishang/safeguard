import random

from datasets import Dataset


class JailBreakExampleRepo:
    _dataset: list[dict[str, str]]

    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset.to_list()

    def get_example(self, prompt_column_name: str = "prompt") -> str:
        return random.choice(self._dataset)[prompt_column_name]
