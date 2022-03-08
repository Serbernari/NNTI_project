import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from utils import similarity_score
import transformer_encoder


"""
Wrapper class using Pytorch nn.Module to create the architecture for our model
Architecture is based on the paper: 
A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING
https://arxiv.org/pdf/1703.03130.pdf
"""


class SiameseBiLSTMAttention(nn.Module):
    def __init__(
        self,
        batch_size,
        output_size,
        hidden_size,
        vocab_size,
        embedding_size,
        embedding_weights,
        lstm_layers,
        device,
        bidirectional,
        self_attention_config,
        attention_encoder_config,
        fc_hidden_size,
        pad_index,
        dropout=0.5, 
    ):
        super(SiameseBiLSTMAttention, self).__init__()
        """
        Initializes model layers and loads pre-trained embeddings from task 1
        """
        ## model hyper parameters
        self.batch_size = batch_size
        self.output_size = output_size
        self.lstm_hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.lstm_layers = lstm_layers
        self.device = device
        self.bidirectional = bidirectional
        self.fc_hidden_size = fc_hidden_size
        self.lstm_directions = (
            2 if self.bidirectional else 1
        )  ## decide directions based on input flag

        ## set dropout
        self.dropout = dropout
        self.self_attention_config = self_attention_config

        ## configuration of hyperparameters for attention encoder blocks
        self.attention_encoder_config = attention_encoder_config

        ## token used for padding to create attention mask
        self.pad_index = pad_index
        
        ## initialize look up table
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)

        ## assign the look-up table to the pre-trained fasttext word embeddings.
        self.word_embeddings.weight = nn.Parameter(embedding_weights.to(self.device), requires_grad=True)

        ## initialize transformer encoder layer
        self.encoder = transformer_encoder.Encoder(
            device, 
            self.attention_encoder_config['n_layers'], 
            dropout, embedding_size, 
            self.attention_encoder_config['n_heads'], 
            self.attention_encoder_config['expansion'], 
            vocab_size, 
            self.attention_encoder_config['vocab_max']
        )

        ## initialize lstm layer
        self.biLSTM= nn.LSTM(
            self.embedding_size,
            self.lstm_hidden_size,
            num_layers=self.lstm_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
        )

        ## incase we are using bi-directional lstm we'd have to take care of bi-directional outputs in
        ## subsequent layers
        ## self attention layer, config come from self_attention_config
        self.attention = SelfAttention(
            self.lstm_hidden_size * self.lstm_directions,
            self.self_attention_config['hidden_size'],
            self.self_attention_config['output_size']
        )

        ## fully connected output layer
        self.fc_layer = nn.Sequential(
            nn.Linear(self.lstm_directions* self.lstm_hidden_size* self_attention_config["output_size"],self.fc_hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.fc_hidden_size, self.output_size, bias=False)
        )


    def init_hidden(self, batch_size):
        """
        Initializes hidden and context weight matrix before each
                forward pass through LSTM
        """
        weight_shape = (self.lstm_layers*self.lstm_directions, batch_size, self.lstm_hidden_size)
        hidden_weights = Variable(torch.zeros(weight_shape).to(self.device))
        context_weights = Variable(torch.zeros(weight_shape).to(self.device))

        return hidden_weights, context_weights
        

    def forward_once(self, batch, attention_encoder_mask):
        """
        Performs the forward pass for each batch
        """

        ## batch shape: (batch_size, seq_len) ([64, 14])
        ## embeddings shape: ( batch_size, seq_len, embedding_size) ([64, 14, 300])

        embeddings = self.word_embeddings(batch)

        ## encoder_out shape: (seq_len, batch_size, embedding_size)
        encoder_out = self.encoder(embeddings, attention_encoder_mask).permute(1,0,2) 

        ## lstm_out = H in AAAI paper
        ## lstm_out shape: (seq_len, batch_size, bidirectional*hidden_size) ([14, 64, 256])
        lstm_out, (fhs, fcs) = self.biLSTM(encoder_out, self.lstm_hidden_weights)#H

        ## go batch first again
        ## shape: (batch_size, seq_len, bidirectional*hidden_size)
        lstm_out = lstm_out.permute(1,0,2)

        ## A matrix from AAAI paper
        attention = self.attention(lstm_out)

        ## out shape: (batch_size, attention_output, bidirectional*hidden_size) ([64, 20, 256])
        out = torch.bmm(attention, lstm_out)

        ## after reshape: (batch_size, attention_output*bidirectional*hidden_size)
        out = out.reshape(-1, self.lstm_directions*self.lstm_hidden_size*self.self_attention_config['output_size'])

        ## final shape after fc layer: (batch_size,output_size)
        out = self.fc_layer(out)

        return out, attention

    def forward(self, sent1_batch, sent2_batch):
        """
        Performs the forward pass for each batch
        """
        ## get multihead attention encoder mask. Mask padding so no attention is paid to pad tokens
        sent1_mask = self.get_encoder_mask(sent1_batch)
        sent2_mask = self.get_encoder_mask(sent2_batch)

        ## init context and hidden weights for lstm cell
        ## batch size might change at the end of an epoch
        self.lstm_hidden_weights = self.init_hidden(sent1_batch.shape[0])

        ## implement forward pass on both sentences. calculate similarity using similarity_score()
        sent_A, sent_A_attention = self.forward_once(sent1_batch, sent1_mask)
        self.sent_A_mh_att_matrix = self.encoder.layer_attention_matrices
        sent_B, sent_B_attention = self.forward_once(sent2_batch, sent2_mask)
        self.sent_B_mh_att_matrix = self.encoder.layer_attention_matrices
        similarity = similarity_score(sent_A, sent_B)

        return similarity, sent_A_attention, sent_B_attention

    def get_encoder_mask(self, input_sentence_batch):
        '''
        Gets mask for the multihead attention encoder layers. This is so <pad> tokens will have no attention paid to them.
        '''
        src_mask = (input_sentence_batch != self.pad_index).unsqueeze(1).unsqueeze(2)
        ## shape: (batch_size, 1, 1, seq_len)

        return src_mask.to(self.device)


class SelfAttention(nn.Module):
    """
    Implementation of the attention block
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(SelfAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.att_output_size = output_size

        ## Ws1 from AAAI
        self.Ws1 = nn.Linear(self.input_size, self.hidden_size, bias=False)
        ## Ws2 from AAAI
        self.Ws2 = nn.Linear(self.hidden_size, self.att_output_size, bias=False)

    ## the forward function would receive lstm's all hidden states as input
    def forward(self, attention_input):
        ## attention_input shape: (batch_sie, seq_len, n_lyer*hidden_size)

        ## x_shape: (batch_size, seq_len, attention_hidden)
        x = self.Ws1(attention_input)

        ## x_shape: (batch_sie, seq_len, attention_hidden)
        x = torch.tanh(x)

        ## x_shape: (batch_size, seq_len, attention_output_size)
        x = self.Ws2(x)

        ## x_shape: (batch_size, attention_output_size, seq_len)
        x = x.permute(0,2,1) 
        
        ## Softmax across dim 2 as written in paper
        return F.softmax(x, dim=2)



