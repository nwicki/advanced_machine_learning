import pandas
import numpy
import test_func
import sklearn
from sklearn import preprocessing

csv_file = 'train.csv'
data = pandas.read_csv(csv_file)
x = numpy.asarray(data.iloc[:, 2:]).astype('float64')
y = numpy.asarray(data['y']).astype('float64')
# (X^T*X)^-1 * X^T * y = b
xt = x.transpose()
b = numpy.matmul(numpy.matmul(numpy.linalg.pinv(numpy.matmul(xt, x)), xt), y)

output = numpy.matmul(test_func.get_test_data(), b)
test_func.test(output, 'direct')
