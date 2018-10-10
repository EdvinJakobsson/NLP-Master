from reader import *
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


data = read_dataset()
print(data[98])

X = [1,2,3,4,5,6,7,8,9,10]
Y = X

#X, Y = shuffle(X, Y, random_state=1)

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.20, random_state=415)

print(train_x)
print(test_x)
print(train_y)
print(test_y)

