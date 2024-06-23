from modules import *

class Guard():
    def __init__(self, fn):
        self.fn = fn
        self.detector = Detector(binary=True)
        self.sanitizer = IterativeSanitizer()
        self.classifier = Classifier()
    
    def __call__(self, inp, classifier=False, sanitizer=False):
        output = {
            "safe": [],
            "class": [],
            "sanitized": [],
        }
        if type(inp) == str:
            inp = [inp]
        vuln = self.detector.forward(inp)
        v = vuln[0]
        # [0 1 1 1 0 0]
        output["safe"].append(v == 0)
        if v == 0:
            output["class"].append('safe input (no classification)')
            output["sanitized"].append('safe input (no sanitization)')
            response = self.fn.forward(inp[0])
        else: # v == 1 -> unsafe case
            if classifier:
                classification = self.classifier.forward(inp)
                output["class"].append(classification)
            if sanitizer:
                sanitized = self.sanitizer.forward(inp)
                output["sanitized"].append(sanitized)
                response = self.fn.forward(sanitized)
            if not sanitizer:
                response = "Sorry, this is detected as a dangerous input."

        return response, output

"""
actual call:

gpt = GPT()
out = gpt(inp)

llm = Guard(llm)

print(llm("what is the meaning of life?"))




"""