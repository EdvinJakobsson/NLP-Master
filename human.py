import reader
from extra_functions import plot_confusion_matrix
import numpy as np
import extraction_functions as extract
import matplotlib.pyplot as plt


def human_raters_agreement_matrix(start, stop, savefile, title="Human Rater Agreement"):
     number_of_essays = stop-start
     data = reader.read_dataset(stop, start)
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
     plot = plot_confusion_matrix(rater1list,rater2list, classes=class_names, title=title)
     plt.savefig(savefile)



savefile = "matr.png"
human_raters_agreement_matrix(999,1248, savefile, "Confusion Matrix")
