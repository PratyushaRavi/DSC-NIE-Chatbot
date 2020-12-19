import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
punctuans = ['!', ',', '$', '.','?',"'"]

def preprocesss_text(sentence):
    tokens =  nltk.wordpunct_tokenize(sentence)                        #tokenisze
    stemmed = [stemmer.stem(word.lower()) for word in tokens]          #stem it
    return [words for words in stemmed if words not in punctuans]


def word_dict(all_words):
    word_dict = {}
    for idx, words in enumerate(all_words):
        word_dict[words] = idx+1
    return word_dict


def bag_of_word(tokenize_sen,wordict):
    bg_token = np.zeros(len((wordict.keys())))
    for idx,token in enumerate(tokenize_sen):
        if token in wordict.keys():
            bg_token[idx] = wordict[token]
        else:
            bg_token[idx]= -1.0     # assign -1 to words not present in allwords
    return bg_token



# senten = "I's anyone there?"
# allwords = ['hey','nice','talk','to','you','ffs','hug']
# prep = (preprocesss_text(senten))
# print(prep)
# word_dct = word_dict(allwords)
# print(word_dct)
# print(bag_of_word(prep,word_dct))