import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from pytorch_transformers import BertModel, GPT2Model
from transformers import GPTJForCausalLM


"""
class TransformerModel(nn.Module):
    def __init__(self, device, config, labels=None):
        super().__init__()

        #if config.model == "BertCased":
            #self.bert = BertModel.from_pretrained('bert-base-cased')
        #elif config.model == 'GPT2':
            #self.bert = GPT2Model.from_pretrained('gpt2')
        #else:
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.fc = nn.Linear(768, labels).to(device)
        self.device = device

    def forward(self, x, y):

        x = x.to(self.device)
        y = y.to(self.device)

        #if self.training:
            #self.bert.train()
            #enc = self.bert(x)[0]
        #else:
        self.bert.eval()
        with torch.no_grad():
            enc = self.bert(x)[0]
        logits = self.fc(enc).to(self.device)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat
"""


class TransformerModel(nn.Module):

    def __init__(self, device, ntoken, d_model, nhead, d_hid, nlayers, config, dropout: float = 0.0, norm_first=True):
        super().__init__()
        self.device = device
        self.config = config
        #if config.model == "BertCased":
            #self.bert = BertModel.from_pretrained('bert-base-cased')
        #elif config.model == 'GPT2':
            #self.bert = GPT2Model.from_pretrained('gpt2')
        #else:
        if config.gpt == 1:
            self.bert = GPT2Model.from_pretrained('gpt2')
        elif config.gpt == 2:
          self.bert = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True, output_hidden_states = True, return_dict=True)#.to("cuda")
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        #self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.lstm = torch.nn.LSTM(input_size=768, hidden_size=768, num_layers=1, batch_first=True)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        #self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def reshapeTensor(ten):
        seqlen = ten.shape[1]
        hdim = ten.shape[2]
        ten = ten.repeat(1,1,2)
        ten = ten.reshape(ten.shape[0], ten.shape[1]*2, hdim)
        ten = ten[:, 1:]
        ten = torch.cat((ten, torch.zeros(ten.shape[0], 1, ten.shape[2])), axis=1)
        ten = ten.reshape(ten.shape[0]*int(ten.shape[1]/2), 2,  ten.shape[2])

    def forward(self, x, y):

        #Args:
        #    src: Tensor, shape [seq_len, batch_size]
        #    src_mask: Tensor, shape [seq_len, seq_len]

        #Returns:
        #    output Tensor of shape [seq_len, batch_size, ntoken]

        x = x.to(self.device)
        y = y.to(self.device)

        #if self.training:
            #self.bert.train()
            #enc = self.bert(x)[0]
        #else:
        self.bert.eval()
        with torch.no_grad():
            if self.config.gpt == 2:
                enc = self.bert(x).hidden_states[28]
                enc.to("cuda")
            else:
                enc = self.bert(x)[0]

        #LSTM
        """
        enc = self.reshapeTensor(enc)
        lstm = self.lstm(enc)[1][0]
        lstm = lstm.reshape(x.shape[0], x.shape[1], x.shape[2])
        """
        ####


        #print(enc.size())
        #src = (self.encoder(enc.long()) * math.sqrt(self.d_model)).to(self.device)
        #src = self.encoder(enc.long()).to(self.device)
        src = self.pos_encoder(enc.permute(1, 0, 2).to(self.device))
        #src = self.pos_encoder(enc.to(self.device))
        #print('src', src.size())
        src_mask = generate_square_subsequent_mask(src.size()[0]).to(self.device)
        #print(src_mask.size())
        output = self.transformer_encoder(src, src_mask)
        output = output.permute(1,0,2)
        #print('output', output.size())
        logits = self.decoder(output)
        #print('logits',logits.size())
        y_hat = logits.argmax(-1)
        #print('y', y.size())
        #print('hat', y_hat.size())

        return logits, y, y_hat








def generate_square_subsequent_mask(sz: int) -> Tensor:
    #Generates an upper-triangular matrix of -inf, with zeros on diag.
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=2)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:

        #Args:
        #    x: Tensor, shape [seq_len, batch_size, embedding_dim]

        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
