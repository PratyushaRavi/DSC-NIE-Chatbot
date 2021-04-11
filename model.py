import torch
import torch.nn as nn
total_words = 100
embed_dim = 7
class NeuralNet(nn.Module):
    def __init__(self,embed_dim,hidden_size,no_class):
        super(NeuralNet, self).__init__()
        self.embed = nn.Embedding(total_words, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size = embed_dim, hidden_size =hidden_size,num_layers = 1, batch_first =True, bidirectional=True)
        self.l1 = nn.Linear(hidden_size *2, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size , no_class)

    def forward(self, x):
        out = self.embed(x)
        lstm_out, (ht, ct) = self.lstm(out)
        hidden = torch.cat((ht[-2, :, :], ht[-1, :, :]), dim=1)
        out = self.l1(hidden)
        out = self.relu(out)
        out = self.l2(out)
        return out
