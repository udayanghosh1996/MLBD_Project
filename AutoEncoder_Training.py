import os
import torch
from torch import nn, optim
from torch.autograd import Variable
from time import time
from AutoEncoder import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(text_reader, epochs):
    if not os.path.exists(os.path.join(os.getcwd(), 'model')):
        os.mkdir(os.path.join(os.getcwd(), 'model'))
    length = 0
    for data in text_reader:
        length += data.shape[0]
    path = os.path.join(os.path.join(os.getcwd(), 'model'), 'AutoEncoder')
    model = AE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-04, eps=1e-08, weight_decay=3e-02)
    criterion = nn.MSELoss()
    time0 = time()
    for e_no in range(1, epochs + 1):
        running_loss = 0
        for data in text_reader:
            data = data.to_numpy(dtype='float32')
            data = Variable(torch.from_numpy(data)).to(device)
            model.train()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            running_loss = loss.item()
        print("Epoch {} - Training Loss for AutoEncoder Training: {}".format(e_no, running_loss / length))
        print("Training Time in Minute=", (time() - time0) / 60)
    torch.save(model, path)
