import csv

def read_dataset(stop):
    #essayfile = r"/home/william/m18_edvin/Projects/Data/asap-aes/training_set_rel3.tsv"
    essayfile = r"C:\Users\Edvin\Projects\NLP-Master\Data\training_set_1.tsv"
    with open(essayfile, newline='', encoding='latin1') as tsvfile:
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
