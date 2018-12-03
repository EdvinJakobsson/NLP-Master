import reader
import extraction_functions as extract
import extra_functions
import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold


def human_raters_agreement(number_of_essays):
     data = reader.read_dataset(number_of_essays)
     rater1 = np.zeros((number_of_essays, 0))
     rater2 = np.zeros((number_of_essays, 0))
     rater1 = extract.score(rater1, 3, data)
     rater2 = extract.score(rater2, 4, data)

     human_raters_agreement = extra_functions.quadratic_weighted_kappa_score(rater1, rater2)
     return human_raters_agreement


def kfold_mlp(number_of_essays, kfold_splits, layer1, layer2, epochs, learning_rate, dropout):

     x = np.zeros((number_of_essays, 0))
     d = np.zeros((number_of_essays, 0))
     data = reader.read_dataset(number_of_essays)

     x = extract.word_length(x, data)
     x = extract.average_word_length(x, data)
     x = extract.stan_dev_word_length(x, data)
     x = extract.dale_score(x, data)

     x = tf.keras.utils.normalize(x, axis=0)

     d = extract.score(d, 6, data)
     d = d-2 #get scores between 0 and 10

     skfold = KFold(n_splits=kfold_splits, shuffle=True)
     kappas = []

     for train_index, test_index in skfold.split(x, d):
          x_train, d_train = x[train_index], d[train_index]
          x_test, d_test = x[test_index], d[test_index]
          train_acc, test_acc, kappa = extra_functions.mlp(x_train, d_train, x_test, d_test, layer1, layer2, epochs, learning_rate, dropout)
          kappas.append(kappa)
          print("Training acc: %2.3f, Test acc: %2.3f, Kappa: %2.3f" % (train_acc, test_acc, kappa))

     average_kappa = sum(kappas) / len(kappas)
     print("Average kappa: %2.3f" % average_kappa)


def main():

     number_of_essays = 1780
     kfold_splits = 5
     layer1 = 128
     layer2 = 128
     epochs = 200
     learning_rate = 0.001
     dropout = 0

     print("Human Agreement Kappa: %2.3f" % human_raters_agreement(number_of_essays))

     kfold_mlp(number_of_essays, kfold_splits, layer1, layer2, epochs, learning_rate, dropout)


if __name__ == "__main__":
     main()
