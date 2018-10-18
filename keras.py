import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from reader import *
from extraction_functions import *

number_of_essays = 1780
x = np.zeros((number_of_essays,1))
y = np.zeros((number_of_essays,0))

data = read_dataset(number_of_essays)
#x = extract_length(x, data)
y = extract_domain1_score(y, data)

x_train = x
y_train = y

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.20, random_state=415)

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(13, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'] )
model.fit(x_train, y_train, epochs=10)


val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

p = model.predict([x_test])

for i in range(len(p)):
    print(np.argmax(p[i]))