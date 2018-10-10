import csv

def read_dataset():

    with open('training_set_1.tsv') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        counter = 0
        data = []
        for row in reader:
            if counter > 0:
                data.append(row)
            counter = counter + 1
            if counter == 100:
                break
    return(data)


# skapa en array av wordCount där talen kan manipuleras
# wordCountArray = np.array([])
# for essay in wordCount:
#     wordCountArray = np.append([wordCountArray], [essay])
#
# print(wordCount)
# print(wordCountArray/2.0)
