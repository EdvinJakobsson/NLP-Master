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
from BenHamner.score import mean_quadratic_weighted_kappa
import time

def human_raters_agreement(number_of_essays):
     data = reader.read_dataset(number_of_essays)
     rater1 = np.zeros((number_of_essays, 0))
     rater2 = np.zeros((number_of_essays, 0))
     rater1 = extract.score(rater1, 3, data)
     rater2 = extract.score(rater2, 4, data)
     human_raters_agreement = extra_functions.quadratic_weighted_kappa_score(rater1, rater2)
     return human_raters_agreement

def human_raters_agreement_matrix(number_of_essays, savefile, title="Human Rater Agreement"):
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






def kfold_mlp(essayset, output, kfold_splits, nodes, layers, learning_rate, dropout, numbers_of_kappa_measurements, epochs_between_kappa, path, essayfile):
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
    loss_ymax = 2
    if essayset == 7:
        loss_ymax = 20
        epochs_between_kappa = epochs_between_kappa*2
    if essayset == 8:
        loss_ymax = 40
        epochs_between_kappa = epochs_between_kappa*2
    if output == 'softmax':
        loss_ymax = 2

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

    i = 0
    for train_index, test_index in skfold.split(x, d):
        i += 1
        print("split ", i)
        x_train, d_train = x[train_index], d[train_index]
        x_test, d_test = x[test_index], d[test_index]

        if output == 'softmax':
            model = models.mlp_softmax(nodes, layers, learning_rate, dropout, number_of_inputs, number_of_classes)
        elif output == 'linear':
            model = models.mlp_linear(nodes, layers, learning_rate, dropout, number_of_inputs)
        elif output == 'sigmoid':
            model = models.mlp_sigmoid(layer1, layer2, learning_rate, dropout, number_of_inputs)

        for step in range(numbers_of_kappa_measurements):
            train_loss, train_acc = model.evaluate(x=x_train, y=d_train, verbose=False, sample_weight=None, steps=None)
            test_loss, test_acc = model.evaluate(x=x_test, y=d_test, verbose=False, sample_weight=None, steps=None)
            train_kappa = extra_functions.quadratic_weighted_kappa_for_MLP(x_train, d_train, essayset, model, output)
            test_kappa = extra_functions.quadratic_weighted_kappa_for_MLP(x_test, d_test, essayset, model, output)

            train_kappa_list[step] += train_kappa
            test_kappa_list[step] += test_kappa
            train_loss_list[step] += train_loss
            test_loss_list[step] += test_loss

            model.fit(x_train, d_train, epochs=epochs_between_kappa, verbose=False)

        train_loss, train_acc = model.evaluate(x=x_train, y=d_train, batch_size=1, verbose=False, sample_weight=None, steps=None)
        test_loss, test_acc = model.evaluate(x=x_test, y=d_test, batch_size=1, verbose=False, sample_weight=None, steps=None)
        train_kappa = extra_functions.quadratic_weighted_kappa_for_MLP(x_train, d_train, essayset, model, output)
        test_kappa = extra_functions.quadratic_weighted_kappa_for_MLP(x_test, d_test, essayset, model, output)

        train_kappa_list[numbers_of_kappa_measurements] += train_kappa
        test_kappa_list[numbers_of_kappa_measurements] += test_kappa
        train_loss_list[numbers_of_kappa_measurements] += train_loss
        test_loss_list[numbers_of_kappa_measurements] += test_loss


    train_kappa_list = [i/kfold_splits for i in train_kappa_list]
    test_kappa_list = [i/kfold_splits for i in test_kappa_list]
    train_loss_list = [i/kfold_splits for i in train_loss_list]
    test_loss_list = [i/kfold_splits for i in test_loss_list]

    plot_kappa = path + "kappa_essayset_" + str(essayset) + ".png"
    extra_functions.plot_kappa(plot_kappa, epoch_list, train_kappa_list, test_kappa_list, title = "Essay set: " + str(essayset), x_axis="Epoch")
    plot_loss = path + "loss_essayset_" + str(essayset) + ".png"
    extra_functions.plot_loss(plot_loss, epoch_list, train_loss_list, test_loss_list, title = "Essay set: " + str(essayset), x_axis="Epoch", y_max = loss_ymax)
    conf_matrix = path + "confusionMatrix_essayset_" + str(essayset) + ".png"
    extra_functions.save_confusion_matrix(conf_matrix, model, x_test, d_test, essayset, output, title="Confusion Matrix")

    return model, test_kappa_list

def main():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    #removes some of the tf warnings
    essayfile = "/home/william/m18_edvin/Projects/Data/asap-aes/training_set_rel3.tsv"

    kappa_file = open("MLP-Results/kappas.txt", "+w")
    kappa_file.write("Nodes per layer \r")
    kfold_splits = 5
    essaysets = [1,2,3,4,5,6,7,8]
    dropout = 0
    epochs = 200
    epochs_between_kappa = 1
    learning_rate = 0.001
    node_models = [50]
    layers = 4
    output = 'softmax'

    essayfile = "C:/Users/Edvin/Projects/Data/asap-aes/training_set_rel3.tsv"
    #kfold_splits = 2
    essaysets = [1]
    epochs = 100

    for nodes in node_models:
        print("nodes per layer: ", nodes)
        kappa_file.write(str(nodes) + "\t \t")
        path = "MLP-Results/" + str(output) + str(layers) + "layers/nodes" + str(nodes) + "/"
        os.makedirs(path)
        kappa_list = []
        for essayset in essaysets:
            print("essayset ", essayset)

            number_of_kappa_measurements = epochs
            if essayset == 7:
                number_of_kappa_measurements = 200
            if essayset == 8:
                number_of_kappa_measurements = 200

            model, test_kappa_list = kfold_mlp(essayset, output, kfold_splits, nodes, layers, learning_rate, dropout, number_of_kappa_measurements, epochs_between_kappa, path, essayfile)
            kappa_list.append(test_kappa_list[-1])
            kappa_file.write(str(test_kappa_list[-1]) + "\t")
        print(kappa_list)
        mean_kappa = mean_quadratic_weighted_kappa(kappa_list)
        print(mean_kappa)
        kappa_file.write("\t" + str(mean_kappa) + "\r")

    kappa_file.close()


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    time = end-start
    hours = int(time/3600)
    min = int((time-3600*hours)/60)
    sec = int(time - hours*3600 - min*60)
    print("Run-Time: " + str(hours) + " hours, " + str(min) + " minutes, " + str(sec) + " seconds.")
    print("done")
