import random
import json
import torch
from model import NeuralNet
from utils import bag_of_word,preprocesss_text
import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
device = 'cuda'

with open ('data.json','r') as f:
    chats = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data['input_size']
output_size = data['output_size']
hidden_dim = data['hidden_dim']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size,hidden_dim,output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = 'sam'
print("Yo chatter box || 'quit' to exit")

while True:
    sentence = input('You: ')
    if sentence == 'quit':
        break
    else:
        sentence = nltk.wordpunct_tokenize(sentence)
        sentence = [stemmer.stem(word.lower()) for word in sentence]
        X = bag_of_word(sentence,all_words)
        X = X.reshape(1,X.shape[0])
        X = torch.from_numpy(X).float().to(device)

        output = model(X)
        _,predicted = torch.max(output,dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output,dim=1)
        prob = probs[0][predicted.item()]
        print(prob)
        if prob > 0.5:
            for intent in chats['intents']:
                if tag == intent["tag"]:
                    print(f'{bot_name}: {random.choice(intent["responses"])}')
        else:
            print(f'{bot_name} : I do not understand...')

