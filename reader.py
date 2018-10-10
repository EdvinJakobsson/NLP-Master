import tensorflow as tf
import csv
import nltk
import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize

a = 0
wordCount = []
essay = []
data = []

with open('training_set_1.tsv') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    for row in reader:
        if a > 0:
            essay.append(row[2])
            data.append(row)
            wordCount.append(len(row[2].split()))
        a = a + 1
        if a == 3:
            break

print(data)

#skapa en array av wordCount där talen kan manipuleras
# wordCountArray = np.array([])
# for essay in wordCount:
#     wordCountArray = np.append([wordCountArray], [essay])
#
# print(wordCount)
# print(wordCountArray/2.0)
