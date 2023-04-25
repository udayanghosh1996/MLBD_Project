from torch import nn


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        # Encoder
        self.enc = nn.Sequential(nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, 16),
                                 nn.Sigmoid()
                                 )
        # Decoder
        self.dec = nn.Sequential(nn.Linear(16, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 512),
                                 nn.Sigmoid())

    def forward(self, data):
        mtx = self.enc(data)
        mtx = self.dec(mtx)
        return mtx

    def cust_forward(self, data):
        mtx = self.enc(data)
        return mtx
