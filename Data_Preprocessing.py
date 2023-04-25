import numpy as np
import pandas as pd
import os
import re

data_dir = os.path.join(os.path.join(os.getcwd(), 'LF-AmazonTitles-1.3M.bow'), 'LF-AmazonTitles-1.3M')


def data_process_create_csv(file):
    data = []
    with open(os.path.join(data_dir, file), 'r') as f:
        Lines = f.readlines()
    for line in Lines:
        data.append(re.split('[; , : ]', line.replace('\n', '')))
    max_col = 0
    if file == 'test.txt':
        max_col = 511
    for row in data:
        if len(row) > max_col:
            max_col = len(row)
    max_row = len(data)
    np_data = np.zeros([max_row, max_col], dtype=np.float64)
    for i, row in enumerate(data):
        for j, col in enumerate(row):
            if col != '':
                np_data[i][j] = col
    df = pd.DataFrame(np_data)
    if not os.path.exists(os.path.join(os.getcwd(), 'data')):
        os.mkdir(os.path.join(os.getcwd(), 'data'))
    df.to_csv(os.path.join(os.path.join(os.getcwd(), 'data'), file.replace('.txt', '.csv')))


if __name__ == "__main__":

    data_process_create_csv('train.txt')

    data_process_create_csv('test.txt')
