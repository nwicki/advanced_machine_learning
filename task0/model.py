import torch.nn as nn
import torch.nn.functional as nnf


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        lin = 10
        l1 = 50
        lout = 1
        self.hl1 = nn.Linear(lin, l1)
        self.hl2 = nn.Linear(l1, lout)

    def forward(self, x):
        x = nnf.relu(self.hl1(x))
        x = self.hl2(x)
        return x
