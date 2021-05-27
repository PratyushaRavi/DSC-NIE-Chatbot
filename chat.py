import random
import json
import torch
import discord
import mysql.connector

from model import NeuralNet
from utils import bag_of_word,preprocesss_text
import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
device = 'cpu'



with open ('data.json','r') as f:
    chats = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)
total_words = data['total_words']
embed_dim = data['embed_dim']
output_size = data['output_size']
hidden_dim = data['hidden_dim']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(total_words,embed_dim,hidden_dim,output_size).to(device)
model.load_state_dict(model_state)
model.eval()


class MyClient(discord.Client):
    async def on_ready(self):
        print('Logged in as')
        print(self.user.name)
        print(self.user.id)
        print('------')

    async def on_message(self, message):
        # we do not want the bot to reply to itself
        if message.author.id == self.user.id:
            return

        else:
            sentence = message.content
            print(sentence)
            sentence = preprocesss_text(sentence)

            X = bag_of_word(sentence, all_words)
            X = X.reshape(1, X.shape[0])
            X = torch.from_numpy(X).to(torch.int64).to(device)

            output = model(X)
            _, predicted = torch.max(output, dim=1)
            tag = tags[predicted.item()]

            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicted.item()]
            # print(prob)
            if prob > 0.5:
                for intent in chats['intents']:
                    if tag == intent["tag"]:
                        await message.channel.send(random.choice(intent["responses"]))
            else:
                await message.channel.send('Sorry,I am not aware about it.My team will get back to you very soon :)')



client = MyClient()
client.run('ODMwNjgyNjIwNTE3ODEwMTk2.YHKPeg.07XdZZpoVO8XVUdMrLy-s-6lH-A')

#
# bot_name = 'DSC Bot'
# print("Yo chatter box || 'quit' to exit")
#
#
# while True:
#     sentence = input('You : ')
#     if sentence == 'quit':
#         break
#     else:
#         sentence = preprocesss_text(sentence)
#
#         X = bag_of_word(sentence,all_words)
#         X = X.reshape(1,X.shape[0])
#         X = torch.from_numpy(X).to(torch.int64).to(device)
#
#         output = model(X)
#         _,predicted = torch.max(output,dim=1)
#         tag = tags[predicted.item()]
#
#         probs = torch.softmax(output,dim=1)
#         prob = probs[0][predicted.item()]
#         # print(prob)
#         if prob > 0.5:
#             for intent in chats['intents']:
#                 if tag == intent["tag"]:
#                     print(f'{bot_name}: {random.choice(intent["responses"])}')
#         else:
#             print(f'{bot_name} : I do not understand...')
