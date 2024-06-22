import argparse
import torch

from modules import Detector, IterativeSanitizer


def main(args):
    # Model
    # TODO: add ability to specify GPU number
    detector = Detector(port_number=args.port)
    sanitizer = IterativeSanitizer()
    classifier = Classifier(port_number=args.port)
   
    while True:
        try:
            inp = input(f"Input a string to detect: ")
            output = detector.forward([inp])
            if output[0][1][0]['label'] == 'INJECTION' and args.enable_sanitizer:
                print("\tDetected prompt injection:")
                print("\t\tOriginal input:\n\t\t" + inp)
                sanitized_inp = sanitizer.forward([inp])
                print("\t\tSanitized input:\n\t\t" + sanitized_inp)
                output = detector.forward([sanitized_inp])
            print(output)

            classification = classifier.forward(inp)
            print(classification)
        except EOFError:
            inp = ""
        except Exception as e:
            print("Exception reached...\n\t" + repr(e))
        if not inp:
            print("exit...")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default="8000")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--enable_sanitizer", action="store_true")
    args = parser.parse_args()
    main(args)