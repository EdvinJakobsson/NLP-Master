from reader import *
from extraction_functions import *
from score import *
import numpy as np


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

