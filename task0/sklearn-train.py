import pandas
import numpy
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import test_func

csv_file = 'train.csv'
data = pandas.read_csv(csv_file)
x = preprocessing.scale(numpy.asarray(data.iloc[:, 2:]).astype('float64'))
y = numpy.asarray(data['y']).astype('float64')

reg = LinearRegression().fit(x, y)
test_func.test(reg.predict(test_func.get_test_data()), 'sklearn-lr')
