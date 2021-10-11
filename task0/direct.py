import pandas
import numpy

csv_file = 'train.csv'
data = pandas.read_csv(csv_file)
x = numpy.asarray(data.iloc[:, 2:]).astype('float32')
y = numpy.asarray(data['y']).astype('float32')
# (X^T*X)^-1 * X^T * y = b
xt = x.transpose()
b = numpy.matmul(numpy.matmul(numpy.linalg.pinv(numpy.matmul(xt, x)), xt), y)
# X * b = y
yhat = numpy.matmul(x, b)

test_data = pandas.read_csv('test.csv')
test_x = numpy.asarray(test_data.iloc[:, 1:]).astype('float32')
output = numpy.matmul(test_x, b)
start = 10000
end = start + len(output)
dataframe = pandas.DataFrame(output, index=range(start, end), columns={'y'})
dataframe.index.name = 'Id'
dataframe.to_csv('submission_direct.csv')


