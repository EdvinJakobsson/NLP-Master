import csv

def read_dataset(stop):

    with open("Data/training_set_1.tsv") as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        counter = 0
        data = []
        for row in reader:
            if counter > 0:
                data.append(row)
            if counter == stop:
                break
            counter = counter + 1
    return(data)



