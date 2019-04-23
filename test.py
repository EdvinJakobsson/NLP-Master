import reader
import extraction_functions as extract
import extra_functions
import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
from keras.utils import to_categorical
import matplotlib.pyplot as plt

def human_raters_agreement(number_of_essays):
     data = reader.read_dataset(number_of_essays)
     rater1 = np.zeros((number_of_essays, 0))
     rater2 = np.zeros((number_of_essays, 0))
     rater1 = extract.score(rater1, 3, data)
     rater2 = extract.score(rater2, 4, data)
     human_raters_agreement = extra_functions.quadratic_weighted_kappa_score(rater1, rater2)
     return human_raters_agreement





def human_raters_agreement_matrix(number_of_essays, savefile, title="human rater agreement"):
     data = reader.read_dataset(number_of_essays)
     rater1 = np.zeros((number_of_essays, 0))
     rater2 = np.zeros((number_of_essays, 0))
     rater1 = extract.score(rater1, 3, data)
     rater2 = extract.score(rater2, 4, data)
     rater1list = []
     rater2list = []
     for i in range(len(rater1)):
         rater1list.append(int(rater1[i]))
         rater2list.append(int(rater2[i]))

     lowest_score = 0
     highest_score = 6
     class_names = np.array(lowest_score)
     for i in range(lowest_score+1,highest_score+1):
         class_names = np.append(class_names, i)
     plot = extra_functions.plot_confusion_matrix(rater1list,rater2list, classes=class_names, title=title)
     plt.savefig(savefile)


def save_confusion_matrix(savefile, model, x, d, lowest_score, highest_score, title=None):
    predictions, targets = argmax(x, d, model)

    class_names = np.array(lowest_score)
    for i in range(lowest_score+1,highest_score+1):
        class_names = np.append(class_names, i)

    plot = plot_confusion_matrix(targets, predictions, classes=class_names,
                      title=title)
    plt.savefig(savefile)

def argmax(x_val, d_val, model):
    p = model.predict([x_val])
    y_test = []
    d_test = []
    for i in range(len(x_val)):
        y_test.append(np.argmax(p[i]))
        d_test.append(np.argmax(d_val[i]))

    return(y_test, d_test)










def kfold_mlp(number_of_essays, kfold_splits, layer1, layer2, epochs, learning_rate, dropout):

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
     x = np.zeros((number_of_essays, 0))
     d = np.zeros((number_of_essays, 0))
     data = reader.read_dataset(number_of_essays)

     x = extract.word_length(x, data)
     #x = extract.average_word_length(x, data)
     #x = extract.stan_dev_word_length(x, data)
     #x = extract.dale_score(x, data)
     x = tf.keras.utils.normalize(x, axis=0)

     d = extract.score(d, 6, data)
     d = d-2 #get scores between 0 and 10

     skfold = KFold(n_splits=kfold_splits, shuffle=True)
     kappas = []
     i = 0

     for train_index, test_index in skfold.split(x, d):
          print("split ", i+1)
          x_train, d_train = x[train_index], d[train_index]
          x_test, d_test = x[test_index], d[test_index]
          train_acc, test_acc, kappa, model = extra_functions.mlp(x_train, d_train, x_test, d_test, layer1, layer2, epochs, learning_rate, dropout)
          kappas.append(kappa)
          print("Training acc: %2.3f, Test acc: %2.3f, Kappa: %2.3f" % (train_acc, test_acc, kappa))

          imagefile = "test" + str(i) + ".png"
          targets = to_categorical(np.asarray(d_test)) #creates a target vector for each text. If a text belongs to class 0 out of 4 classes the vector will be: [1., 0., 0., 0.]
          #extra_functions.save_confusion_matrix(imagefile, model, x_test, targets, 2, 12, title=None)
          i += 1
     average_kappa = sum(kappas) / len(kappas)
     print("Average kappa: %2.3f" % average_kappa)


def main():

     number_of_essays = 1246
     kfold_splits = 5
     layer1 = 20
     layer2 = 20
     epochs = 200
     learning_rate = 0.001
     dropout = 0.99


     # number_of_essays = 10
     # kfold_splits = 2
     # layer1 = 2
     # layer2 = 2
     # epochs = 2
     # learning_rate = 0.001
     # dropout = 0

     print("Human Agreement Kappa: %2.3f" % human_raters_agreement(number_of_essays))

     kfold_mlp(number_of_essays, kfold_splits, layer1, layer2, epochs, learning_rate, dropout)


if __name__ == "__main__":
    for i in range(1):
        print(i)
        main()

        #number_of_essays = 1246
        #savefile = "hej.png"
        #human_raters_agreement_matrix(number_of_essays, savefile, title="Human Rater Agreement, set 1, 1246 essays")
        print("done")
