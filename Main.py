import os
import pandas as pd
from AutoEncoder_Training import *
from Generate_Embeddings import *

if __name__ == "__main__":

    train_reader = pd.read_csv(os.path.join(os.path.join(os.getcwd(), 'data'), 'train.csv'), chunksize=10000)

    test_reader = pd.read_csv(os.path.join(os.path.join(os.getcwd(), 'data'), 'test.csv'), chunksize=10000)

    train(train_reader, 50)

    generate_embedding_pickle(test_reader, 'Autoencoder')
