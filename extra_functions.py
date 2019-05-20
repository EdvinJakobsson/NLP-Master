from reader import *
import extraction_functions as extract
import numpy as np
import tensorflow as tf
from collections import Counter
import matplotlib.pyplot as plt
import reader_full
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from BenHamner.score import quadratic_weighted_kappa, mean_quadratic_weighted_kappa


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


def plot_loss(filename, epochs, train_loss, val_loss, title, x_axis, y_max = 2):
    plt.plot(epochs,train_loss, "r--", label='Training Loss')
    plt.plot(epochs,val_loss, label='Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel(x_axis)
    plt.ylim(0,y_max)
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()


def quadratic_weighted_kappa_score(d_array, y_array):
    d = []
    y = []
    for i in range(len(d_array)):
        d.append(int(d_array[i]))
        y.append(int(y_array[i]))

    kappa = quadratic_weighted_kappa(d, y)

    return kappa




def quadratic_weighted_kappa_for_MLP(x_val, d_val, essayset, model, output):
    y_test, d_test = argmax(x_val, d_val, essayset, model, output)
    kappa = quadratic_weighted_kappa(d_test, y_test)
    return kappa




def argmax(x_val, d_val, essayset, model, output):
    asap_ranges = {
    0: (0, 60),
    1: (2, 12),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (0, 30),
    8: (0, 60)
    }
    max_score = asap_ranges[essayset][1] - asap_ranges[essayset][0]
    p = model.predict([x_val])
    predictions = []
    targets = []
    if output == 'softmax':
        for i in range(len(x_val)):
            predictions.append(np.argmax(p[i]))
            targets.append(int(d_val[i]))
    elif(output == 'sigmoid'):
        for i in range(len(x_val)):
            predictions.append(int(p[i]*max_score+0.5))
            targets.append(int(d_val[i]*max_score))
    elif(output == 'linear'):
        for i in range(len(x_val)):
            targets.append(int(d_val[i]))
            prediction = int(p[i]+0.5)
            if prediction > max_score:
                prediction = max_score
            if prediction < 0:
                prediction = 0
            predictions.append(prediction)
    else:
        print("argmax: something wrong with 'output' value")
    return(predictions, targets)


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






def save_confusion_matrix(savefile, model, x, d, essayset, output, title=None):
    asap_ranges = {0: (0, 60), 1: (2, 12), 2: (1, 6), 3: (0, 3), 4: (0, 3), 5: (0, 4), 6: (0, 4), 7: (0, 30), 8: (0, 60) }
    lowest_score = asap_ranges[essayset][0]
    highest_score = asap_ranges[essayset][1]
    predictions, targets = argmax(x, d, essayset, model, output)
    class_names = np.array(lowest_score)
    for i in range(lowest_score+1,highest_score+1):
        class_names = np.append(class_names, i)

    plot = plot_confusion_matrix(targets, predictions, classes=class_names, title=title)
    plt.savefig(savefile)





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
