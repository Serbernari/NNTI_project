import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import similarity_score, cosine_similarity
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
        vocab_size,
        embedding_size,
        device,
        attention_encoder_config,
        pad_index,
        dropout=0.5, 
    ):
        super(SiameseBiLSTMAttention, self).__init__()
        """
        Initializes model layers and loads pre-trained embeddings from task 1
        """
        ## model hyper parameters
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.device = device
        self.dropout = dropout

        ## configuration of hyperparameters for attention encoder blocks
        self.attention_encoder_config = attention_encoder_config

        ## token used for padding to create attention mask
        self.pad_index = pad_index
        
        ## Variable embedding depth, traiing from scratch
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)

        #print('word_embeddings shape is {}'.format(self.word_embeddings.weight.shape))
        self.encoder = transformer_encoder.Encoder(
            device, 
            self.attention_encoder_config['n_layers'], 
            dropout, embedding_size, 
            self.attention_encoder_config['n_heads'], 
            self.attention_encoder_config['expansion'], 
            vocab_size, 
            self.attention_encoder_config['vocab_max']
        )

        #bidirectional lstm layer
        # self.biLSTM= nn.LSTM(
        #     self.embedding_size,
        #     self.lstm_hidden_size,
        #     num_layers=self.lstm_layers,
        #     bidirectional=self.bidirectional,
        #     dropout=self.dropout,
        # )

        ## incase we are using bi-directional lstm we'd have to take care of bi-directional outputs in
        ## subsequent layers
        #self attention layer, config come from self_attention_config
        # self.attention = SelfAttention(
        #     self.lstm_hidden_size * self.lstm_directions,
        #     self.self_attention_config['hidden_size'],
        #     self.self_attention_config['output_size']
        # )

        ## fully connected output layer
        # self.fc_layer = nn.Sequential(
        #     nn.Linear(self.lstm_directions* self.lstm_hidden_size* self_attention_config["output_size"],self.fc_hidden_size, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(self.fc_hidden_size, self.output_size, bias=False)
        # )


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
        #print('batch shape is {}'.format(batch.shape))
        ## embeddings shape: ( batch_size, seq_len, embedding_size) ([64, 14, 300])
        #print('embeddings shape is {}'.format(self.word_embeddings(batch).shape))
        embeddings = self.word_embeddings(batch)#.permute(1,0,2)
        #print('embeddings shape is {}'.format(self.word_embeddings(batch).shape)) #([64, 14, 300])

        encoder_out = self.encoder(embeddings, attention_encoder_mask)#.permute(1,0,2) #(seq_len, batch_size, embedding_size)
        #print('encoder_out shape is {}'.format(encoder_out.shape))

        encoder_out = self.mean_pool(encoder_out, attention_encoder_mask)

        # print('lstm_hidden_weights is {}'.format(self.lstm_hidden_weights))
        #lstm_out, (fhs, fcs) = self.biLSTM(encoder_out, self.lstm_hidden_weights)#H

        # out shape (seq_len, batch_size, bidirectional*hidden_size)([14, 64, 256])
        #print('lstm output shape is {}'.format(lstm_out.shape))

        #go batch first again
        #lstm_out = lstm_out.permute(1,0,2)

        #attention = self.attention(encoder_out) #A
        #print('attention shape is {}'.format(attention.shape))

        #out = torch.bmm(attention, encoder_out)
        #(batch_size, attention_output, bidirectional*hidden_size) ([64, 20, 256])
        #print('after bmm shape is {}'.format(out.shape))

        #(batch_size, attention_output*bidirectional*hidden_size)
        #out = out.reshape(-1, self.lstm_directions*self.lstm_hidden_size*self.self_attention_config['output_size'])
        #print('after reshape output shape is {}'.format(out.shape))
        #out = self.fc_layer(encoder_out)
        #(batch_size,output_size)
        #print('after fully connected shape is {}'.format(out.shape))

        return encoder_out#, attention

    def forward(self, sent1_batch, sent2_batch):
        """
        Performs the forward pass for each batch
        """
        ## batch size might change toward the end of epoch
        #self.lstm_hidden_weights = self.init_hidden(sent1_batch.shape[0])

        ## mask for input sentence padding indices
        sent1_mask = self.get_encoder_mask(sent1_batch)
        sent2_mask = self.get_encoder_mask(sent2_batch)

        ## Send each batch through the siamese network 
        sent_A = self.forward_once(sent1_batch, sent1_mask)
        sent_B = self.forward_once(sent2_batch, sent2_mask)

        ## Calculate embedding similarity
        similarity = cosine_similarity(sent_A, sent_B) #similarity_score(sent_A, sent_B)
        #print('similarity is {}'.format(similarity.shape))

        return similarity#, sent_A_attention, sent_B_attention

    def get_encoder_mask(self, input_sentence_batch):
        src_mask = (input_sentence_batch != self.pad_index).unsqueeze(1).unsqueeze(2)
        #shape: (batch_size, 1, 1, sen_len)

        return src_mask.to(self.device)

    def mean_pool(self, embeddings, mask):
        ## take only relevant dimensions from mask (batch_size, 1, 1, sen_len) => (batch_size, sen_len)
        mask = mask[: ,0 ,0,:]

        ## expand to embedding size (batch_size, sen_len) => (batch_size, sen_len, embedding_size)
        mask = mask.unsqueeze(-1).expand(embeddings.size()).float()

        ## set all embeddings that belong to <pad> tokens to 0
        masked = embeddings * mask

        ## sum values along sequence_len dimension
        summed_embeddings = torch.sum(masked, 1)

        ## get number of non padded entries for sentences
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)

        mean = summed_embeddings / summed_mask
        return mean


class SelfAttention(nn.Module):
    """
    Implementation of the attention block
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(SelfAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.att_output_size = output_size

        ## Ws1 from paper
        self.Ws1 = nn.Linear(self.input_size, self.hidden_size, bias=False)
        ## Ws2 from paper
        self.Ws2 = nn.Linear(self.hidden_size, self.att_output_size, bias=False)

    ## the forward function would receive lstm's all hidden states as input
    def forward(self, attention_input):
        #([64, 14, 256]) (batch_sie, seq_len, n_lyer*hidden_size)
        #print('attention_iput has shape has shape{}'.format(attention_input.shape))

        x = self.Ws1(attention_input)
        #([64, 14, 150]) (batch_sie, seq_len, attention_hidden)
        #print('after first pass x has shape{}'.format(x.shape))

        x = torch.tanh(x)
        #([64, 14, 150]) (batch_sie, seq_len, attention_hidden)
        #print('after tanh x has shape{}'.format(x.shape))

        x = self.Ws2(x)
        #([64, 14, 20]) (batch_sie, seq_len, attention_output_size)
        #print('after second pass x has shape{}'.format(x.shape))

        x = x.permute(0,2,1) #(batch_size, attention_output_size, seq_len)
        return F.softmax(x, dim=2)



