from numpy.core.fromnumeric import size
import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadSelfAttention
from torch.nn.modules.container import ModuleList
from torch.nn.modules.sparse import Embedding

class EncoderBlock(nn.Module):
    def __init__(self, dropout_p, size_embed, n_heads, expansion):
        super(EncoderBlock, self).__init__()

        self.attention = MultiHeadSelfAttention(size_embed, n_heads)
        self.Dropout = nn.Dropout(dropout_p)

        #Layer normalisation taken from pytorch
        self.LayerNorm1 = nn.LayerNorm(size_embed)
        self.LayerNorm2 = nn.LayerNorm(size_embed)

        #feed forward network to get final result of the block
        #They used a fully connected layer, ReLU activation and another fully connected layer
        #The fully connected layers expand the input by the expansion factor, then reduce it again
        self.FeedForward = nn.Sequential(
            nn.Linear(size_embed, size_embed*expansion),
            nn.ReLU(),
            nn.Linear(size_embed*expansion, size_embed)
        )

    def forward(self, Q,K,V, mask=None):
        #dropout attention
        attention = self.Dropout(self.attention(Q,K,V,mask))
        #skipped connection before normalisation
        add = attention+Q
        #The architecture has some normalisation
        int_res = self.LayerNorm1(add)
        #put intermediate result through feed forward network
        ff = self.Dropout(self.FeedForward(int_res))

        #add int_res to ff output and use normalisation
        return self.LayerNorm2(ff+int_res)





class Encoder(nn.Module):
    def __init__(self, device, n_layers, dropout_p, size_embed, n_heads, expansion, src_vocab, vocab_max):
        super(Encoder, self).__init__()
        self.device = device
        self.layers = self.make_layers(n_layers, dropout_p, size_embed, n_heads, expansion)
        self.size_embed = size_embed
        self.dropout_p = dropout_p
        self.n_heads = n_heads
        self.expansion = expansion
        self.src_vocab = src_vocab

        ## positional embedding encodes word position within sentences
        self.pos_embedding = nn.Embedding(vocab_max, size_embed)
        self.Dropout = nn.Dropout(dropout_p)


    def forward(self, input_embedding, mask):
        ## shape: (batch_size, sequence_len, embedding_size)
        n, seq_len, embedding_size = input_embedding.shape
        
        #get positions of words in input to make embeddings. Add positional embeding to word embedding to get 
        #positional information into the embeddings
        input_positions = torch.arange(0, seq_len).expand(n, seq_len).to(self.device)

        ## input is already an embedding using fastvec thing
        b = self.pos_embedding(input_positions)
        embedd = input_embedding + b
        res = self.Dropout(embedd)
        res = input_embedding

        #repeat the internal encoder loop n_layers times
        for layer in self.layers:
            res = layer(res, res, res, mask)

        return res

    def make_layers(self, n_layers, dropout_p, size_embed, n_heads, expansion):
        return nn.ModuleList([EncoderBlock(dropout_p, size_embed, n_heads, expansion) for i in range(n_layers)])



