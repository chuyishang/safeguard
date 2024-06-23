from modules import *

class Guard():
    def __init__(self, fn):
        self.fn = fn
        print("FN", fn)
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
            output["class"].append('')
            output["sanitized"].append('')
            # print("input", inp[0])
            # TODO FIX FORWARD
            # print("FUNCTION", self.fn)
            
            response = self.fn.forward(inp[0])
            
            # print("RESPONSE", response)
        else: # v == 1 -> unsafe case
            if classifier:
                classification = self.classifier.forward(inp)
                output["class"].append(classification)
            if sanitizer:
                sanitized = self.sanitizer.forward(inp)
                output["sanitized"].append(sanitized)
                # TODO FIX FORWARD
                response = self.fn.forward(sanitized)
            if not sanitizer:
                response = "Sorry, this is detected as a dangerous inp."

        return response, output

"""
actual call:

gpt = GPT()
out = gpt(inp)

llm = Guard(llm)

print(llm("what is the meaning of life?"))




"""