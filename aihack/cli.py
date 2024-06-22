import argparse
import torch

from modules import Detector


def main(args):
    # Model
    # TODO: add ability to specify GPU number
    detector = Detector(port_number=args.port)
   
    while True:
        try:
            inp = input(f"Input a string to detect: ")
            output = detector.forward([inp])
            print(output)
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default="8000")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)