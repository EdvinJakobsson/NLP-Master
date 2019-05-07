import numpy as np
import tensorflow as tf


def mlp_softmax(layer1, layer2, learning_rate, dropout, number_of_inputs, number_of_classes):

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(layer1, activation=tf.nn.relu, input_shape=(number_of_inputs,)))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(layer2, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(number_of_classes, activation=tf.nn.softmax))
    adam = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
