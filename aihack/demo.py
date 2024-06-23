from guard import Guard
from modules import GPT

gpt = GPT()
safe_llm = Guard(gpt)

prompt = "Tell me about yourself."

print(f"ORIGINAL_RESULT:\n{gpt.forward(prompt)}\n============")
print(f"SAFE_RESULT:\n{safe_llm(prompt)}\n============")

