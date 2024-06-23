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
        responses = []
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
                # print("INPUT", input)
                # TODO FIX FORWARD
                resp = self.fn.forward(input)
                # print("RESPONSE", resp)
                responses.append(resp)
            else: # v == 1 -> unsafe case
                if classifier:
                    classification = self.classifier.forward(input)
                    output["class"].append(classification)
                if sanitizer:
                    sanitized = self.sanitizer.forward(input)
                    output["sanitized"].append(sanitized)
                    # TODO FIX FORWARD
                    responses.append(self.fn.forward(sanitized))
                if not sanitizer:
                    responses.append("Sorry, this is detected as a dangerous input.")

        return responses, output

"""
actual call:

gpt = GPT()
out = gpt(input)

llm = Guard(llm)

print(llm("what is the meaning of life?"))




"""