import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from reader import *
from extraction_functions import *
import matplotlib.pyplot as plt
from extra_functions import *

number_of_essays = 1780
x = np.zeros((number_of_essays, 0))
y = np.zeros((number_of_essays, 0))

data = read_dataset(number_of_essays)
x = extract_word_length(x, data)
x = extract_average_word_length(x, data)
y = extract_score(y, 6, data)
y = make_onehotvector(y, 2, 12)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=415)

#x_train = tf.keras.utils.normalize(x_train, axis=1)
#x_test = tf.keras.utils.normalize(x_test, axis=1)


model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(20, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(11, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'] )
estimator = model.fit(x_train, y_train, epochs=200, validation_data=(x_test, y_test))



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

d = []
for i in range(len(x_test)):
    d.append(np.argmax(p[i]))

print("result: {}".format(d))
stats(d)