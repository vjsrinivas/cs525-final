import numpy as np
import os
import yaml
import argparse

if __name__ == '__main__':
    print("")
    parser = argparse.ArgumentParser()
    parser.add_argument('--np', type=str)
    parser.add_argument('--data', type=str)
    args = parser.parse_args()
    
    data = np.load(args.np, allow_pickle=True)
    with open(args.data, "r") as stream:
        try:
            dataset = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    classes = dataset.names
    print(data, classes)