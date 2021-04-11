from torch.utils.data import DataLoader,Dataset
import json
import torch
from utils import preprocesss_text,word_dict,bag_of_word
from asd import NeuralNet

with open ('data.json','r') as f:
    chats = json.load(f)

tags=[]
all_words =[]
train = []

for idx,intent in enumerate(chats['intents']):
    tags.append(intent['tag'])
    for pattern in intent['patterns']:
        pattern = preprocesss_text(pattern)
        all_words.extend(pattern)
        all_words = list(set(all_words))
        train.append((pattern,idx))

x_train = []
y_train = []
allwords_dictionary = word_dict(all_words)
for sentence,clas in train:
    x_train.append(bag_of_word(sentence,allwords_dictionary))
    y_train.append(clas)





class ChatDataset(Dataset):
    def __init__(self):
        self.no_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.no_samples

#############Hyperparameters############
batch_size = 16
embed_dim = 64
hidden_dim = 16
no_class = len(tags)
device = 'cuda'
lr = 0.002
no_epoch = 500

dataset = ChatDataset()
train_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

model = NeuralNet(embed_dim,hidden_dim,no_class).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer =  torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

for epoch in range(no_epoch):
    for (words,label) in train_loader:
        words = words.to(torch.int64).to(device)
        labels = label.to(device)
        # print(type(words))
        #forward pass
        output = model(words)
        loss = criterion(output,labels)

        #backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1)%100 == 0:
              print(f'epoch {epoch+1}/{no_epoch}, loss = {loss.item():.4f}')

print(f'final loss , loss={loss.item():.4f}')

data ={
    "model_state":model.state_dict(),
    "embed_dim":embed_dim,
    "output_size":no_class,
    "hidden_dim": hidden_dim,
    "all_words":allwords_dictionary,
    "tags":tags
}

FILE = "data.pth"
torch.save(data,FILE)

print(f'Training complete aand file saved to {FILE}')
