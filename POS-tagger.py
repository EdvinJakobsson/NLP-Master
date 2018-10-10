import csv
import nltk
from nltk.tokenize import PunktSentenceTokenizer

essay = []
wordCount = []
all_essays = ''


#read in data
with open('training_set_1.tsv') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    counter = 0
    for row in reader:
        if counter > 0:
            essay.append(row[2])
            wordCount.append(len(row[2].split()))
            all_essays += ' ' + row[2]
        counter += 1
        if counter == 3:
            break

##creating an object, custom trained POS-tagger
#custom_sent_tokenizer = PunktSentenceTokenizer(all_essays)

##creating a list of the sentences (strings)
#tokenized = custom_sent_tokenizer.tokenize(essay[1])

tokenized = nltk.sent_tokenize(essay[0])

def pos_tagger():
    try:
        for i in tokenized:

            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)

    except Exception as e:
        print(str(e))

pos_tagger()
