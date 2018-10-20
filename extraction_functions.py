from reader import *
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt


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

# ändra så att row[x] är variabel, kalla på funktionen från
#  andra funktioner, snarare än att kopiera den här 15 ggr

##kolla också fördelningen av betyg!

def extract_domain1_score(array, data):
    try:
        values = []
        for row in data:
            values.append(float(row[6]))

        array = np.c_[array, values]
    except:
        print("extract_domain1_score function error.")
        exit(1)


    return array


#data = read_dataset(10)
#array = np.zeros((10, 0))
#array = extract_length(array, data)
#print(array)


# Display the distribution of grades
def stats():
    data = read_dataset(1780)
    d = np.zeros((1780, 0))
    d = extract_domain1_score(d, data)
    plt.hist(d)
    #plt.yscale("log")
    plt.show()

