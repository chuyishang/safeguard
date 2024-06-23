import json
import time

from modules import Detector

start_time = time.time()

with open("/home/shang/aihack/aihack/data/malicious_data.json", "r") as file:
    malicious_data = json.load(file)
full = []
for row in malicious_data:
    prompt = row["example_jailbreak_input"]
    full.append(prompt)

detector = Detector(binary=True)
outputs = detector.forward(full)

correct = 0
for output in outputs:
    correct += output


print(correct / len(outputs))
print(correct, len(outputs))

end_time = time.time()
print(f"Total time taken: {end_time - start_time} seconds")
