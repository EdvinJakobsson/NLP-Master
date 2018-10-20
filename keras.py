import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from reader import *
from extraction_functions import *
import matplotlib.pyplot as plt

number_of_essays = 1780
x = np.zeros((number_of_essays, 0))
y = np.zeros((number_of_essays, 0))

data = read_dataset(number_of_essays)
x = extract_length(x, data)
y = extract_domain1_score(y, data)

x_train = x
y_train = y

d = np.zeros([number_of_essays, 11])
for i in range(number_of_essays):
    d[i, int(y_train[i]-2) ] = 1

y_train = d

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.20, random_state=415)

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

print(x_train.shape)

model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(20, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(11, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'] )
estimator = model.fit(x_train, y_train, epochs=40, validation_data=(x_test, y_test))



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







#val_loss, val_acc = model.evaluate(x_test, y_test)
#print(val_loss, val_acc)

p = model.predict([x_test])

#print(p[0])

#for i in range(len(p)):
#    print(np.argmax(p[i]))

print((np.argmax(p)!=6))