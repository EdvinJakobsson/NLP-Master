from reader import *
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer


def extract_length(array, data):    # array is input values, data is the list created by reader.py.
                                    # The array needs same number of rows as the data (length of data string)
    try:
        # object used to remove punctuations in essay
        tokenizer = RegexpTokenizer(r'\w+')

        values = []
        for row in data:
            values.append(len(tokenizer.tokenize(row[2])))

        array = np.c_[array, values]
    except:
        print("extract_length function error.")
        exit(1)

    return array

##################################################


def extract_domain1_score(array, data):
    try:
        values = []
        for row in data:
            values.append(row[6])

        array = np.c_[array, values]
    except:
        print("extract_domain1_score function error.")
        exit(1)


    return array


#data = read_dataset(10)
#array = np.zeros((10, 0))
#array = extract_domain1_score(array, data)
#print(array)





