from reader import *
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import csv
from sklearn.model_selection import train_test_split
import array as arr

#data = read_dataset()

#X = (1,2,3,4,5,6,7,8,9,10)
#Y = X
#X, Y = shuffle(X, Y, random_state=1)
#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=415)

x_train = np.array([[[1.0]],[[2.0]],[[3.0]],[[4.0]]])
y_train = np.array([1, 2, 3, 4])

x_train, y_train = shuffle(x_train, y_train, random_state=1)

print(x_train)
print(y_train)

print(isinstance(x_train, np.ndarray))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'] )
#model.fit(x_train, y_train, epochs=3000)


