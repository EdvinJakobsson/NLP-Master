import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from reader import *
from extraction_functions import *
import matplotlib.pyplot as plt
from extra_functions import *
from score import *


number_of_essays = 20
x = np.zeros((number_of_essays, 0))
y = np.zeros((number_of_essays, 0))

data = read_dataset(number_of_essays)
x = extract_word_length(x, data)
x = extract_average_word_length(x, data)
x = extract_stan_dev_word_length(x, data)
x = extract_dale_score(x, data)

y = extract_score(y, 6, data)
y = y-2

x = tf.keras.utils.normalize(x, axis=0)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=415)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(11, activation=tf.nn.softmax))

#adam = tf.keras.optimizers.Adam(lr=0.0003)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'] )
estimator = model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), verbose=True)



#####################################

# Evaluate model.
loss, acc = model.evaluate(x=x_train, y=y_train, batch_size=1, verbose=1, sample_weight=None, steps=None)
print('Training loss = {}'.format(loss))
print('Training accuracy = {}'.format(acc))

loss, acc = model.evaluate(x=x_test, y=y_test, batch_size=1, verbose=1, sample_weight=None, steps=None)
print('Test loss = {}'.format(loss))
print('Test accuracy = {}'.format(acc))


# Plot the training error
plt.plot(estimator.history['loss'], label='Training')
plt.plot(estimator.history['val_loss'], label='Validation')
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('loss-function.pdf', bbox_inches='tight')
plt.show()

# Plot the training error¬
plt.plot(estimator.history['acc'], label='Training')
plt.plot(estimator.history['val_acc'], label='Validation')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('accuracy-function.pdf', bbox_inches='tight')
plt.show()

################################


p = model.predict([x_test])
print("Predict 6 for all x: {}".format(np.argmax(p)== 6))

y = []
for i in range(len(x_test)):
    y.append(np.argmax(p[i]))

score = quadratic_weighted_kappa_score(y_test, y)
print("Quadratic weighted Kappa Score: {}".format(score))

#print("result: {}".format(d))
#stats(d)