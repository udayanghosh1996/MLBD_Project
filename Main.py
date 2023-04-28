import os
import pandas as pd
from AutoEncoder_Training import *
from Generate_Embeddings import *
from time import time 
import psutil

if __name__ == "__main__":

    train_reader = pd.read_csv(os.path.join(os.path.join(os.getcwd(), 'data'), 'train.csv'), chunksize=10000)

    test_reader = pd.read_csv(os.path.join(os.path.join(os.getcwd(), 'data'), 'test.csv'), chunksize=10000)

    train(train_reader, 50)
    start = time()
    generate_embedding_pickle(test_reader, 'Autoencoder')
    end = time()
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
    print('The CPU usage is: ', psutil.cpu_percent(4))
    print('Time taken for Generate embeddings =',(end - start) /60, 'min')
