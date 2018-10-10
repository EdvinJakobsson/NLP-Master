from reader import *
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#read in the dataset as a list of lists (rows, columns)
data = read_dataset()


X = [[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7,],[8,8],[9,9,],[10,10]]
Y = [[1],[2],[3],[4],[5],[6],[7,],[8],[9,],[10]]

#Shuffle the dataset to mix up the rows
X, Y = shuffle(X, Y, random_state=1)


#Split the dataset into training and testing
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.20, random_state=415)


#Define important variables
learning_rate = 0.3
training_epochs = 100
n_dim = len(X[0])
n_class = 11

#### dubbel - backslash ??????!!!!!!!!   \\
model_path = "C:/Users/Edvin/PycharmProjects/untitled/NLP-Master/models/mlp"

n_hidden_1 = 3
n_hidden_2 = 3
n_hidden_3 = 3
n_hidden_4 = 3

x = tf.placeholder(tf.float32,[None, n_dim])
W = tf.Variable(tf.zeros([n_dim, n_class]))
b = tf.Variable(tf.zeros([n_class]))
y_ = tf.placeholder(tf.float32, [None, n_class])

#Define the model
def multilayer_perceptron(x, weights, biases):
    #Hidden layer with RELU activations

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    #Output layer with linear activation
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer

#Define the wréights and the biases for each layer
weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_class])),
}

biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_class])),
}

#Initialize all the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()

#call your model
y = multilayer_perceptron(x, weights, biases)

cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

sess = tf.Session()
sess.run(init)

# Calculate the cost and the accuracy for each epoch

mse_history = []
accuracy_history = []

sess.run(training_step, feed_dict={x: train_x, y_: train_y})
