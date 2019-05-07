from reader import *
from extraction_functions import *
from BenHamner.score import quadratic_weighted_kappa
import numpy as np
import tensorflow as tf
from collections import Counter
import matplotlib.pyplot as plt

def plot_kappa(filename, epochs, train_kappa, val_kappa, title, x_axis):
    plt.plot(epochs,train_kappa, "r--", label='Training Kappa')
    plt.plot(epochs,val_kappa, label='Validation Kappa')
    plt.ylabel('Kappa')
    plt.xlabel(x_axis)
    plt.ylim(-0.1,1)
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()


def plot_loss(filename, epochs, train_kappa, val_kappa, title, x_axis):
    plt.plot(epochs,train_kappa, "r--", label='Training Loss')
    plt.plot(epochs,val_kappa, label='Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel(x_axis)
    plt.ylim(0,3)
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()

def mlp(x_train, d_train, x_test, d_test, layer1, layer2, epochs, learning_rate, dropout, number_of_classes):

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(layer1, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(layer2, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(number_of_classes, activation=tf.nn.softmax))
    adam = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    model.fit(x_train, d_train, epochs=epochs, verbose=False)
    loss, train_acc = model.evaluate(x=x_train, y=d_train, batch_size=1, verbose=False, sample_weight=None, steps=None)
    loss, test_acc = model.evaluate(x=x_test, y=d_test, batch_size=1, verbose=False, sample_weight=None, steps=None)

    p = model.predict([x_test])
    y_test = []
    for i in range(len(x_test)):
        y_test.append(np.argmax(p[i]))


    kappa_score = quadratic_weighted_kappa_score(d_test, y_test)

    return train_acc, test_acc, kappa_score, model



def quadratic_weighted_kappa_score(d_array, y_array):
    d = []
    y = []
    for i in range(len(d_array)):
        d.append(int(d_array[i]))
        y.append(int(y_array[i]))

    kappa = quadratic_weighted_kappa(d, y)

    return kappa

def kappa(model, x, d):
    p = model.predict([x])
    y = []
    for i in range(len(x)):
        y.append(np.argmax(p[i]))

    kappa_score = quadratic_weighted_kappa_score(d, y)

    return kappa_score


def make_onehotvector(array, lowest_grade, highest_grade):
    number_of_essays = len(array)
    number_of_grades = highest_grade - lowest_grade + 1
    d = np.zeros([number_of_essays, number_of_grades])

    for i in range(number_of_essays):
        d[i, int(array[i] - 2)] = 1

    return d


# Displays the distribution of grades
def stats(d):
    plt.hist(d)
    #plt.yscale("log")
    plt.show()

    #prints number of essays whith each grade
    number_of_essays = 1780
    data = read_dataset(number_of_essays)
    y = np.zeros((number_of_essays, 0))
    y = extract_score(y, 6, data)
    list = y.tolist()
    a = sum(list, [])
    print(Counter(a))





def argmax(x_val, d_val, model):
    p = model.predict([x_val])
    y_test = []
    d_test = []
    for i in range(len(x_val)):
        y_test.append(np.argmax(p[i]))
        d_test.append(np.argmax(d_val[i]))

    return(y_test, d_test)



def save_confusion_matrix(savefile, model, x, d, lowest_score, highest_score, title=None):
    predictions, targets = argmax(x, d, model)

    class_names = np.array(lowest_score)
    for i in range(lowest_score+1,highest_score+1):
        class_names = np.append(class_names, i)

    plot = plot_confusion_matrix(targets, predictions, classes=class_names,
                      title=title)
    plt.savefig(savefile)



import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
