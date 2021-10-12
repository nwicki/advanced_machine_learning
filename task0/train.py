import sys
import pandas
import numpy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as opt
import model
from sklearn import preprocessing


class RandomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pandas.read_csv(csv_file)
        self.x = numpy.asarray(self.data.iloc[:, 2:]).astype('float64')
        self.y = numpy.asarray(self.data['y']).astype('float64').reshape(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        s = {'x': self.x[idx], 'y': self.y[idx]}
        return s


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# epochs = int(sys.argv[1])
batch_size = 10
num_workers = 10
starting_lr = 1e-2
learning_rate = starting_lr
decay_lr = 2
reset = 30
ratio = 1e2
improv_eps = 0.1
final_loss = 1e6
dataset = RandomDataset("train.csv")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
net = model.Net()
criterion = nn.MSELoss()
optimizer = opt.Adam(net.parameters(), lr=learning_rate)
print_freq = len(dataset) / batch_size
net.to(device)
count = 0
target = 1e-15
acceptable = 1e-09
epoch = 0
# for epoch in range(epochs):
while target < final_loss:
    epoch = epoch + 1
    running_loss = 0.0
    for i, sample in enumerate(dataloader):
        input = sample['x'].to(device)
        output = sample['y'].to(device)
        optimizer.zero_grad()
        prediction = net(input)
        loss = criterion(prediction, output)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % print_freq == print_freq - 1:
            final_loss = running_loss / print_freq
            print('[%d] loss: %.3e, learning rate: %.0e' % (epoch, final_loss, learning_rate))
            running_loss = 0.0

    count = count + 1
    if final_loss < learning_rate * ratio:
        count = 0
        learning_rate /= decay_lr
        optimizer = opt.Adam(net.parameters(), lr=learning_rate)
    if reset < count:
        if final_loss < acceptable:
            torch.save(net.state_dict(), "model_%.3e" % final_loss)
        net.__init__()
        criterion = nn.MSELoss()
        net.to(device)
        count = 0
        learning_rate = starting_lr
        optimizer = opt.Adam(net.parameters(), lr=learning_rate)
torch.save(net.state_dict(), "model_%.3e" % final_loss)
