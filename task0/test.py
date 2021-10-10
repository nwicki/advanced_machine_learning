import model
import sys
import torch
import pandas
import numpy

net = model.Net()
net.load_state_dict(torch.load(sys.argv[1]))
data = pandas.read_csv("test.csv")
inputs = numpy.asarray(data.iloc[:, 1:]).astype('float32')
output = net(torch.from_numpy(inputs))
start = 10000
end = start + len(output)
dataframe = pandas.DataFrame(output.detach().numpy(), index=range(start, end), columns={'y'})
dataframe.index.name = 'Id'
dataframe.to_csv('submission_%s.csv' % sys.argv[1])
