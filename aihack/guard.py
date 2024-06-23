from modules import *

class Guard():
    def __init__(self, fn):
        self.fn = fn
        print("FN", fn)
        self.detector = Detector(binary=True)
        self.sanitizer = IterativeSanitizer()
        self.classifier = Classifier()
    
    def __call__(self, input, classifier=False, sanitizer=False):
        output = {
            "safe": [],
            "class": [],
            "sanitized": [],
        }
        if type(input) == str:
            input = [input]
        vuln = self.detector.forward(input)
        v = vuln[0]
        # [0 1 1 1 0 0]
        output["safe"].append(v == 0)
        if v == 0:
            output["class"].append('')
            output["sanitized"].append('')
            print("INPUT", input)
            # TODO FIX FORWARD
            print("FUNCTION", self.fn)
            
            if hasattr(self.fn, 'forward'):
                response = self.fn.forward(input)
            else:
                response = "Function does not support forward method."
            
            response = self.fn.forward(input)
            
            print("RESPONSE", response)
        else: # v == 1 -> unsafe case
            if classifier:
                classification = self.classifier.forward(input)
                output["class"].append(classification)
            if sanitizer:
                sanitized = self.sanitizer.forward(input)
                output["sanitized"].append(sanitized)
                # TODO FIX FORWARD
                response = self.fn.forward(sanitized)
            if not sanitizer:
                response = "Sorry, this is detected as a dangerous input."

        return response, output

"""
actual call:

gpt = GPT()
out = gpt(input)

llm = Guard(llm)

print(llm("what is the meaning of life?"))




"""