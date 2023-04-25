import pickle
import torch
import os
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_embeddings(text_reader, model_name):
    path = os.path.join(os.path.join(os.getcwd(), 'model'), model_name)
    model = torch.load(path).to(device)
    model.eval()
    val_output_data = np.zeros(16)
    for data in text_reader:
        data = data.to_numpy(dtype='float32')
        data = torch.from_numpy(data).to(device)
        val_output = model.cust_forward(data)
        if torch.cuda.is_available():
            val_output = val_output.cpu()
        val_output_data = np.vstack((val_output_data, val_output.detach().numpy()))
    return val_output_data[1:]


def generate_embedding_pickle(text_reader, model_name):
    if not os.path.exists(os.path.join(os.getcwd(), 'embeddings')):
        os.mkdir(os.path.join(os.getcwd(), 'embeddings'))
    path = os.path.join(os.getcwd(), 'embeddings')
    with open(os.path.join(path, 'Embeddings.pkl'), 'wb') as fh:
        pickle.dump(get_embeddings(text_reader, model_name), fh)
