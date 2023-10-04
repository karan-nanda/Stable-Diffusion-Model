import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embed: int, n_token: int):
        super().__init__()
        
        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        
        #Learnable weight matrix that ENCODES the position information for each token
        self.position_embedding = nn.Parameter(torch.zeros((n_token, n_embed)))
    
    def forward(self, tokens):
        
        x = self.token_embedding()
        