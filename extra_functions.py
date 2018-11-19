from reader import *
from extraction_functions import *
from score import *
import numpy as np
import tensorflow as tf
from collections import Counter

def mlp(x_train, d_train, x_test, d_test, epochs=200, dropout=0):

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(28, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(28, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(11, activation=tf.nn.softmax))

    # adam = tf.keras.optimizers.Adam(lr=0.0003)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, d_train, epochs=epochs, verbose=False)

    loss, train_acc = model.evaluate(x=x_train, y=d_train, batch_size=1, verbose=False, sample_weight=None, steps=None)
    loss, test_acc = model.evaluate(x=x_test, y=d_test, batch_size=1, verbose=False, sample_weight=None, steps=None)

    p = model.predict([x_test])
    y_test = []
    for i in range(len(x_test)):
        y_test.append(np.argmax(p[i]))

    kappa_score = quadratic_weighted_kappa_score(d_test, y_test)

    return train_acc, test_acc, kappa_score


def quadratic_weighted_kappa_score(d_array, y_array):
    d = []
    y = []
    for i in range(len(d_array)):
        d.append(int(d_array[i]))
        y.append(int(y_array[i]))

    score = quadratic_weighted_kappa(d, y)

    return score


def make_onehotvector(array, lowest_grade, highest_grade):
    number_of_essays = len(array)
    number_of_grades = highest_grade - lowest_grade + 1
    d = np.zeros([number_of_essays, number_of_grades])

    for i in range(number_of_essays):
        d[i, int(array[i] - 2)] = 1

    return d


# Displays the distribution of grades
def stats(d):
    plt.hist(d)
    #plt.yscale("log")
    plt.show()

    #prints number of essays whith each grade
    number_of_essays = 1780
    data = read_dataset(number_of_essays)
    y = np.zeros((number_of_essays, 0))
    y = extract_score(y, 6, data)
    list = y.tolist()
    a = sum(list, [])
    print(Counter(a))

