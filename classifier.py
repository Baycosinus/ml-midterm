import sys
import pandas as pd
import random_forest as rf
import xg_boost as xgb

data_attributes = []
data_labels = []

def main():
    global dataset
    args = sys.argv # Reading dataset files as arguments
    if (len(args) > 2):
        print("Initializing dataset...")
        data_attributes = pd.read_csv(args[1])
        data_labels = pd.read_csv(args[2])
        print("Dataset initialize successful.")
        print("Dataset shape: " + str(data_attributes.shape))

        #CLASSIFCATION#
        rf.process(data_attributes, data_labels)
        xgb.process(data_attributes, data_labels)
    else:
        print("Dataset argument not found. Please enter dataset filepath and try again.")
        print("USAGE: py classifier.py <ATTRIBUTE DATASET PATH> <LABEL DATASET PATH>")

if (__name__ == "__main__"):
    main()