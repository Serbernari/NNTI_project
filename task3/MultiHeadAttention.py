import torch
import torch.nn as nn
import numpy as np

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, size_emb, num_heads):
        super(MultiHeadSelfAttention, self).__init__()

        self.size_emb = size_emb
        self.num_heads = num_heads
        self.heads_size = size_emb // num_heads

        #Query, Value, Keys are all individual linear layers
        #self.Q = nn.Linear(self.heads_size, self.heads_size)
        #self.K = nn.Linear(self.heads_size, self.heads_size)
        #self.V = nn.Linear(self.heads_size, self.heads_size)

        self.Q = nn.Linear(size_emb, size_emb, bias=False)
        self.K = nn.Linear(size_emb, size_emb, bias=False)
        self.V = nn.Linear(size_emb, size_emb, bias=False)


        self.output_layer = nn.Linear(self.num_heads * self.heads_size, size_emb)
        


    def forward(self, Q, K, V, mask=None):
        #save for later
        Q_len = Q.shape[1]
        K_len = K.shape[1]
        V_len = V.shape[1]
        n = Q.shape[0] #batch size
        #print("Q shape is {}", Q.shape)
        #print("K shape is {}", K.shape)
        #print("V shape is {}", V.shape)
        
        Q,K,V = self.split_qvk(Q,K,V,Q_len,K_len,V_len,n) # shape is (n, len, n_heads, heads_size)

        attention = self.calculate_attention(Q,K,V,Q_len,K_len,V_len,n, mask)
        #print(attention.shape)
        return self.output_layer(attention)



    def split_qvk(self, Q,K,V,Q_len,K_len,V_len,n):
        #linear layers with own weights
        Q = self.Q(Q)
        K = self.K(K)
        V = self.V(V)

        #split attention for multiheaded attention. Go from (n, len, embed_size) to (n, len, n_heads, heads_size)
        Q = Q.reshape(n, Q_len, self.num_heads, self.heads_size)
        #print(K.shape)
        K = K.reshape(n, K_len, self.num_heads, self.heads_size)
        V = V.reshape(n, V_len, self.num_heads, self.heads_size)

    
        return Q,K,V

    def calculate_attention(self,Q,K,V,Q_len,K_len,V_len,n, mask=None):
        #Multiply Q,K as shown in paper
        qK = torch.einsum("nQhs,nKhs -> nhQK", [Q,K]) #shape is (n, n_heads, Q_len, K_len)
        if mask is not None:
            ## To make sure that no attention is paid to the padding
            qK = qK.masked_fill(mask == 0, -1e9)

        #softmax of product, divide by sqrt of embedding size as shown in paper
        s = torch.softmax((qK/(self.size_emb**(1/2))), dim=3)

        #K_len is eqal to V_len so we multiply along this axis
        attention = torch.einsum("nhQK, nKhs->nQhs", [s, V]).reshape(n, Q_len, self.num_heads*self.heads_size)


        return attention