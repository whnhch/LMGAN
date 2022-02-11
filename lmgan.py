import torch
import torch.nn as nn
from transformers import BertModel

class Generator(torch.nn.Module):
    def __init__(self, hidden_size, dkp):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(1-dkp)
        self.activation = nn.LeakyReLU()

        
        self.linear_out = nn.utils.spectral_norm(nn.Linear(hidden_size, hidden_size))
        self.bn = nn.BatchNorm1d(hidden_size)

    def forward(self, z):

        hidden_layer = z[:,0,:]
 
        hidden_layer = self.linear_out(hidden_layer)
        hidden_layer = self.activation(hidden_layer)
        hidden_layer = self.dropout(hidden_layer)

        return hidden_layer

class Discriminator(torch.nn.Module):
    def __init__(self, x_size, hidden_size, dkp, num_labels):
        super().__init__()

        self.dropout = nn.Dropout(1-dkp)
        self.linear =nn.utils.spectral_norm(nn.Linear(x_size, hidden_size))

        self.activation = nn.LeakyReLU()

        self.dense = nn.utils.spectral_norm(nn.Linear(hidden_size, (num_labels)))

        self.cls = nn.Softmax(dim=1)

    def forward(self, x=None):
        hidden_layer = self.dropout(x)

        hidden_layer = self.linear(hidden_layer)
        hidden_layer = self.activation(hidden_layer)
        hidden_layer = self.dropout(hidden_layer)

        logits = self.dense(hidden_layer)
        probs = self.cls(logits)
        
        return hidden_layer, logits, probs

class Classifier(torch.nn.Module):
    def __init__(self, num_labels):
        super().__init__()

        self.dense = nn.Linear(768, (num_labels-1))
        self.cls = nn.Softmax(dim=1)

    def forward(self, x=None):

        logits = self.dense(x)
        probs = self.cls(logits)
        
        return x, logits, probs

class Transformer(torch.nn.Module):
    def __init__(self, transformer_config):
        super().__init__()
        self.transformer =  BertModel(transformer_config).from_pretrained('bert-base-uncased')

    def forward(self, input_ids=None, input_masks=None):

        ret_dict = self.transformer(input_ids=input_ids, attention_mask=input_masks, return_dict=True)
        last_hidden_layer = ret_dict['last_hidden_state']

        pool = last_hidden_layer[:,0,:]

        return pool
