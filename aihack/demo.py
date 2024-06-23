from guard import Guard
from modules import GPT

gpt = GPT()
print(gpt("who are you?"))

safe_llm = Guard(gpt.__call__)

print(safe_llm("who are you?"))

