import reader_full
import extraction_functions as extract
import extra_functions
import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import models
import os


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




def kfold_mlp(essayset, kfold_splits, layer1, layer2, learning_rate, dropout, numbers_of_kappa_measurements, epochs_between_kappa, path, essayfile):
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

    data = reader_full.read_dataset([essayset], filepath=essayfile)
    data = data[:int(len(data)*0.7)]    # save 30% of essays for final evaluation
    number_of_essays = len(data)

    x = np.zeros((number_of_essays, 0))
    d = np.zeros((number_of_essays, 0))

    x = extract.word_length(x, data)
    x = extract.average_word_length(x, data)
    x = extract.stan_dev_word_length(x, data)
    x = extract.dale_score(x, data)
    x = tf.keras.utils.normalize(x, axis=0)
    number_of_inputs = len(x[0])
    d = extract.score(d, 6, data)
    min_score = asap_ranges[essayset][0]
    max_score = asap_ranges[essayset][1]
    number_of_classes = max_score - min_score + 1
    d = d-min_score



    skfold = KFold(n_splits=kfold_splits, shuffle=True)

    epoch_list = [i*epochs_between_kappa for i in range(numbers_of_kappa_measurements+1)]
    train_kappa_list = [0 for i in range(numbers_of_kappa_measurements+1)]
    test_kappa_list = [0 for i in range(numbers_of_kappa_measurements+1)]
    train_loss_list = [0 for i in range(numbers_of_kappa_measurements+1)]
    test_loss_list = [0 for i in range(numbers_of_kappa_measurements+1)]

    for train_index, test_index in skfold.split(x, d):
        x_train, d_train = x[train_index], d[train_index]
        x_test, d_test = x[test_index], d[test_index]

        model = models.mlp_softmax(layer1, layer2, learning_rate, dropout, number_of_inputs, number_of_classes)

        for step in range(numbers_of_kappa_measurements):
            train_loss, train_acc = model.evaluate(x=x_train, y=d_train, verbose=False, sample_weight=None, steps=None)
            test_loss, test_acc = model.evaluate(x=x_test, y=d_test, verbose=False, sample_weight=None, steps=None)
            train_kappa = extra_functions.kappa(model, x_train, d_train)
            test_kappa = extra_functions.kappa(model, x_test, d_test)

            train_kappa_list[step] += train_kappa
            test_kappa_list[step] += test_kappa
            train_loss_list[step] += train_loss
            test_loss_list[step] += test_loss

            model.fit(x_train, d_train, epochs=epochs_between_kappa, verbose=False)

        train_loss, train_acc = model.evaluate(x=x_train, y=d_train, batch_size=1, verbose=False, sample_weight=None, steps=None)
        test_loss, test_acc = model.evaluate(x=x_test, y=d_test, batch_size=1, verbose=False, sample_weight=None, steps=None)
        train_kappa = extra_functions.kappa(model, x_train, d_train)
        test_kappa = extra_functions.kappa(model, x_test, d_test)

        train_kappa_list[numbers_of_kappa_measurements] += train_kappa
        test_kappa_list[numbers_of_kappa_measurements] += test_kappa
        train_loss_list[numbers_of_kappa_measurements] += train_loss
        test_loss_list[numbers_of_kappa_measurements] += test_loss


    train_kappa_list = [i/kfold_splits for i in train_kappa_list]
    test_kappa_list = [i/kfold_splits for i in test_kappa_list]
    train_loss_list = [i/kfold_splits for i in train_loss_list]
    test_loss_list = [i/kfold_splits for i in test_loss_list]

    plot_kappa = path + "kappa/lr" + str(learning_rate) + ".png"
    extra_functions.plot_kappa(plot_kappa, epoch_list, train_kappa_list, test_kappa_list, title = "Dropout: " + str(dropout), x_axis="Epoch")
    plot_loss = path + "loss/lr" + str(learning_rate) + ".png"
    extra_functions.plot_loss(plot_loss, epoch_list, train_loss_list, test_loss_list, title = "Dropout: " + str(dropout), x_axis="Epoch")

    return model

def main():

     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    #removes some of the tf warnings
     essayfile = "/home/william/m18_edvin/Projects/Data/asap-aes/training_set_rel3.tsv"
     kfold_splits = 5
     layer1 = 120
     layer2 = 120
     dropout = 0
     numbers_of_kappa_measurements = 300
     epochs_between_kappa = 1



     # essayfile = "C:/Users/Edvin/Projects/Data/asap-aes/training_set_rel3.tsv"
     # kfold_splits = 2
     # layer1 = 2
     # layer2 = 2

     essaysets = [1,2,3,4,5,6,7,8]
     for essayset in essaysets:
         path = "MLP-Results/essayset" + str(essayset) + "/"
         os.makedirs(path + "kappa/")
         os.makedirs(path + "loss/")
         learning_rates = [0.001, 0.005, 0.01, 0.015, 0.03, 0.05, 0.1]
         for learning_rate in learning_rates:
             print("essayset: ", essayset)
             print("learning_rate: ", learning_rate)



             numbers_of_kappa_measurements = 1
             model = kfold_mlp(essayset, kfold_splits, layer1, layer2, learning_rate, dropout, numbers_of_kappa_measurements, epochs_between_kappa, path, essayfile)



if __name__ == "__main__":
    for i in range(1):
        print(i)
        main()

        #number_of_essays = 1246
        #savefile = "hej.png"
        #human_raters_agreement_matrix(number_of_essays, savefile, title="Human Rater Agreement, set 1, 1246 essays")
        print("done")
