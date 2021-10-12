import torch.nn as nn
import torch.nn.functional as nnf


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        lin = 10
        lout = 1
        hl = int(lin * 2/3 + lout)
        self.hli = nn.Linear(lin, hl).double()
        self.hl1 = nn.Linear(hl, hl).double()
        self.hlo = nn.Linear(hl, lout).double()

    def forward(self, x):
        x = nnf.relu(self.hli(x))
        x = nnf.relu(self.hl1(x))
        x = self.hlo(x)
        return x
