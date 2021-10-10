import sys
import pandas
import numpy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as opt
import model


class RandomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pandas.read_csv(csv_file)
        self.x = numpy.asarray(self.data.iloc[:, 2:]).astype('float32')
        self.y = numpy.asarray(self.data['y']).astype('float32').reshape(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        s = {'x': self.x[idx], 'y': self.y[idx]}
        return s


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = int(sys.argv[1])
batch_size = 10
num_workers = 10
learning_rate = 0.001
dataset = RandomDataset("train.csv")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
net = model.Net()
criterion = nn.MSELoss()
optimizer = opt.Adam(net.parameters(), lr=learning_rate)
print_freq = len(dataset) / batch_size
net.to(device)

final_loss = 0
for epoch in range(epochs):
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
            print('[%d/%d] loss: %.3f' % (epoch + 1, epochs, final_loss))
            running_loss = 0.0
    if epoch % 10 == 0 and epoch != 0:
        learning_rate /= 10
        optimizer = opt.Adam(net.parameters(), lr=learning_rate)
torch.save(net.state_dict(), "model_%.5f" % final_loss)
