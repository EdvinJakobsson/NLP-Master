from reader import *
from score import *
from extraction_functions import *
from extra_functions import *
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# number_of_essays = 10
# data = read_dataset(number_of_essays)
# y = np.zeros((number_of_essays, 0))
# y = extract_score(y, 6, data)
# y = y-2
# d = y
#
# #a = quadratic_weighted_kappa_score(d,y)

d = numpy.array(["a","b","c","d","e,","f","g"])
dd = numpy.array(["A","B","C","D","E,","F","G"])
kfold = KFold(n_splits=7, shuffle=True)

for train_index, test_index in kfold.split(d,d):
    x_train, y_train = d[train_index], dd[train_index]
    print(x_train, y_train)