import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
import re


stemmer = PorterStemmer()
punctuans = ['!', ',', '$', '.','?',"'"]

def emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def preprocesss_text(sent):
    tokens =  nltk.wordpunct_tokenize(sent)                        #tokenisze
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
            bg_token[idx]= 0     # assign 0 to words not present in allwords
    return bg_token




# allwords = ['hey','nice','talk','to','you','ffs','hug']
# senten = "Hi 🤔 How is your 🙈 and 😌. Have a nice weekend �"
# prep = (preprocesss_text(senten))
# print(prep)
# word_dct = word_dict(allwords)
# print(word_dct)
# print(bag_of_word(prep,word_dct))
