import os
import pandas as pd
from pandas import read_csv
import numpy as np

if __name__ == '__main__':
    train_path = os.path.join(os.getcwd(), 'train.csv')
    test_path = os.path.join(os.getcwd(), 'test.csv')
    testdf = read_csv(train_path)
    traindf = read_csv(test_path)
    print(traindf)