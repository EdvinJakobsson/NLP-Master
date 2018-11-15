from reader import *
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
import math
import textstat


def extract_words(text):
    # object used to remove punctuations in essay
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)

    return words

def extract_word_length(array, data):    # array is input values, data is the list created by reader.py.
                                    # The array needs same number of rows as the data (length of data string)
    try:
        values = []
        for row in data:
            values.append(len(extract_words(row[2])))

        array = np.c_[array, values]
    except:
        print("extract_word_length function error.")
        exit(1)

    return array


def extract_score(array, column, data):
    try:
        values = []
        for row in data:
            values.append(float(row[column]))

        array = np.c_[array, values]
    except:
        print("extract_score function error.")
        exit(1)

    return array


def extract_average_word_length(array, data):
    try:
        values = []
        for row in data:
            words = extract_words(row[2])
            average_length = sum(map(len, words)) / len(words)
            values.append(average_length)

        array = np.c_[array, values]
    except:
        print("extract_average_word_length function error.")
        exit(1)

    return array



def extract_stan_dev_word_length(array, data):
    try:
        values = []
        for row in data:
            words = extract_words(row[2])
            average_length = sum(map(len, words)) / len(words)
            average = 0
            for word in words:
                average = average + pow((len(word) - average_length), 2)

            average = average / len(words)
            values.append(math.sqrt(average))

        array = np.c_[array, values]
    except:
        print("extract_stan_dev_word_length function error.")
        exit(1)

    return array


def extract_dale_score(array, data):
    values = []
    for row in data:
        values.append(textstat.dale_chall_readability_score(row[2]))

    array = np.c_[array, values]

    return array


def extract_sentence_length(array, data):
    values = []
    for row in data:
        values.append(textstat.sentence_count((row[2])))

    array = np.c_[array, values]

    return array




# number_of_essays = 20
# data = read_dataset(number_of_essays)
# array = np.zeros((number_of_essays, 0))
# print(array)
# array = extract_word_length(array, data)
# print(array)
# array = extract_average_word_length(array, data)
# print(array)
# array = extract_stan_dev_word_length(array, data)
# array = extract_sentence_length(array, data)
# print(array)
# print(array)
# array = extract_dale_score(array, data)
# print(array)
# array = extract_score(array, 6, data)
# print(array)

