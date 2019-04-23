import numpy as np
import tensorflow as tf

def mlp(layer1=20, layer2=20, learning_rate=0.0003, dropout=0):

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(layer1, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(layer2, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(11, activation=tf.nn.softmax))
    adam = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

model = mlp()
print("done")
