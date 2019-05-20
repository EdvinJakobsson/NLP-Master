import matplotlib.pyplot as plt
def plot(filename, x, y_list, lines, labels, title, x_axis, y_axis):

    for i in range(len(y_list)):
        plt.plot(x, y_list[i], lines[i], label=labels[i])
    plt.ylabel(y_axis)
    plt.xlabel(x_axis)
    plt.ylim(-0.1,1)
    plt.xticks(x, x)
    #plt.xticks(x, x, rotation='vertical')
    plt.subplots_adjust(bottom=0.15)
    plt.margins(0.2)
    plt.title(title)
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig(filename)
    plt.close()



list1 = [
0.318501159,
0.402998829,
0.45908069,
0.499033605]

list2 = [
0.597940122,
0.617502058,
0.627175265,
0.630327977]

list3 = [
0.586336987,
0.652829853,
0.655059859,
0.658851095]

x = ["20","50", "100", "200"]


y_list = []
y_list.append(list1)
y_list.append(list2)
y_list.append(list3)

lines = ['--', '-', '-.']
labels = [ "One Hidden Layer","Two Hidden Layers", "Three Hidden Layers" ]

filename = "testing.png"
title = "Multilayer Perceptron"
x_axis = "Hidden nodes in each layer"
y_axis = "Mean Kappa over all essay sets"


#
# labels = [ "" ]
# number_of_features = [0.621, 0.635, 0.647, 0.652]
# #lines = ["ro"]
# feature = [0.621, 0.020, 0.070 ,0.024]
# #x_axis = ""
# list1 = number_of_features
# x = ["1","2","3","4"]
# x_axis = "Number of handcrafted features"
# #x = ["N. of W.","Avg. L. of W.", "Sta. dev.", "Dale-Chall"]
# y_list = []
# y_list.append(list1)



plot(filename, x, y_list, lines, labels, title, x_axis, y_axis)
