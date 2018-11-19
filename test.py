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
d = np.zeros((number_of_essays, 0))

x = extract_word_length(x, data)
x = extract_average_word_length(x, data)
x = extract_stan_dev_word_length(x, data)
x = extract_dale_score(x, data)
x = tf.keras.utils.normalize(x, axis=0)

d = extract_score(d, 6, data)
d = d-2

rater1 = np.zeros((number_of_essays, 0))
rater2 = np.zeros((number_of_essays, 0))
rater1 = extract_score(rater1, 3, data)
rater2 = extract_score(rater2, 4, data)

human_raters_agreement = quadratic_weighted_kappa_score(rater1, rater2)
print("Human Agreement Kappa: ", human_raters_agreement)

skfold = KFold(n_splits=5, shuffle=True)
kappas = []
for train_index, test_index in skfold.split(x, d):
     x_train, d_train = x[train_index], d[train_index]
     x_test, d_test = x[test_index], d[test_index]
     train_acc, test_acc, kappa = mlp(x_train, d_train, x_test, d_test, 200, 0)
     kappas.append(kappa)
     print("Training acc: ", train_acc, "   Test acc: ", test_acc, "   Kappa: ", kappa)


print("Average kappa: ", sum(kappas) / len(kappas))