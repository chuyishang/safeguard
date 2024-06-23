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
        response = []
        if type(input) == str:
            input = [input]
        
        vuln = self.detector.forward(input)
        print(vuln)
        # [0 1 1 1 0 0]
        for v in vuln:
            output["safe"].append(v)
            if v == 0:
                output["class"].append('')
                output["sanitized"].append('')
                print("INPUT", input)
                response = self.fn(input)
                print("RESPONSE", response)
                response.append(response)
            else: # v == 1 -> unsafe case
                if classifier:
                    classification = self.classifier.forward(input)
                    output["class"].append(classification)
                if sanitizer:
                    sanitized = self.sanitizer.forward(input)
                    output["sanitized"].append(sanitized)
                    response.append(self.fn(sanitized))
                if not sanitizer:
                    response.append("Sorry, this is detected as a dangerous input.")

        return response, output

"""
actual call:

gpt = GPT()
out = gpt(input)

llm = Guard(llm)

print(llm("what is the meaning of life?"))




"""