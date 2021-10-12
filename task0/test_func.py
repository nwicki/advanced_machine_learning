import pandas
import numpy
import sklearn


def get_test_data():
    test_data = pandas.read_csv('test.csv')
    return numpy.asarray(test_data.iloc[:, 1:]).astype('float64')


def test(output, name):
    start = 10000
    end = start + len(output)
    dataframe = pandas.DataFrame(output, index=range(start, end), columns={'y'})
    dataframe.index.name = 'Id'
    dataframe.to_csv('submission_%s.csv' % name)
