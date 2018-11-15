from reader import *
from extraction_functions import *
from score import *
import numpy as np
import tensorflow as tf

def mlp(x_train, d_train, x_test, d_test, epochs=200, dropout=0):
    x_train = tf.keras.utils.normalize(x_train, axis=0)
    x_test = tf.keras.utils.normalize(x_test, axis=0)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(11, activation=tf.nn.softmax))

    # adam = tf.keras.optimizers.Adam(lr=0.0003)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    estimator = model.fit(x_train, d_train, epochs=epochs, verbose=False)


    p = model.predict([x_test])
    y_test = []
    for i in range(len(x_test)):
        y_test.append(np.argmax(p[i]))

    kappa_score = quadratic_weighted_kappa_score(d_test, y_test)

    return kappa_score

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

