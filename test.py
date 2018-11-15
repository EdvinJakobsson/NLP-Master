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
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import itertools

number_of_essays = 1780
data = read_dataset(number_of_essays)
x = np.zeros((number_of_essays, 0))
y = np.zeros((number_of_essays, 0))

x = extract_word_length(x, data)
x = extract_average_word_length(x, data)
x = extract_stan_dev_word_length(x, data)
x = extract_dale_score(x, data)

y = extract_score(y, 6, data)
y = y-2


skfold = StratifiedKFold(n_splits=2)

for train_index, test_index in skfold.split(y, y):
     x_train, d_train = y[train_index], y[train_index]
     x_test, d_test = y[test_index], y[test_index]
     #print(train_index)

     score = mlp(x_train, d_train, x_test, d_test, 200, 0)
     print(score)