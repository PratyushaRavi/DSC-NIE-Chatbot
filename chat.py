import random
import json
import torch

# import mysql.connector
# import streamlit as st
from asd import NeuralNet
from utils import bag_of_word,preprocesss_text
import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
device = 'cpu'

# mydb = mysql.connector.connect(host = 'localhost', user = 'root', passwd = 'EkMzopLL1928!$')

# st.title('DSC Chatbot')

with open ('data.json','r') as f:
    chats = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

embed_dim = data['embed_dim']
output_size = data['output_size']
hidden_dim = data['hidden_dim']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(embed_dim,hidden_dim,output_size).to(device)
model.load_state_dict(model_state)
model.eval()
# st.markdown("Hi DSC Bot here")
bot_name = 'DSC Bot'
print("Yo chatter box || 'quit' to exit")


while True:
    sentence = input('You : ')
    if sentence == 'quit':
        break
    else:
        sentence = preprocesss_text(sentence)

        X = bag_of_word(sentence,all_words)
        X = X.reshape(1,X.shape[0])
        X = torch.from_numpy(X).to(torch.int64).to(device)

        output = model(X)
        _,predicted = torch.max(output,dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output,dim=1)
        prob = probs[0][predicted.item()]
        # print(prob)
        if prob > 0.5:
            for intent in chats['intents']:
                if tag == intent["tag"]:
                    print(f'{bot_name}: {random.choice(intent["responses"])}')
        else:
            print(f'{bot_name} : I do not understand...')

