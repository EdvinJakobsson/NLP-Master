import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, BatchNormalization, Activation
from keras.models import Model, Sequential
from keras.initializers import Constant
from keras import optimizers, regularizers



def mlp_softmax(nodes, layers, learning_rate, dropout, number_of_inputs, number_of_classes):

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(nodes, activation=tf.nn.relu, input_shape=(number_of_inputs,)))
    model.add(tf.keras.layers.Dropout(dropout))
    for i in range(1,layers):
        model.add(tf.keras.layers.Dense(nodes, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(number_of_classes, activation=tf.nn.softmax))
    adam = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def mlp_linear(nodes, layers, learning_rate, dropout, number_of_inputs):

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(nodes, activation=tf.nn.relu, input_shape=(number_of_inputs,)))
    model.add(tf.keras.layers.Dropout(dropout))
    for i in range(1,layers):
        model.add(tf.keras.layers.Dense(nodes, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    return model
