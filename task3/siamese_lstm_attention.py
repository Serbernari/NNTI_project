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
        embedding_weights,
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

        ## assign the look-up table to the pre-trained fasttext word embeddings.
        self.word_embeddings.weight = nn.Parameter(embedding_weights.to(self.device), requires_grad=True)

        ## initialize the transformer encoder layers
        self.encoder = transformer_encoder.Encoder(
            device, 
            self.attention_encoder_config['n_layers'], 
            dropout, embedding_size, 
            self.attention_encoder_config['n_heads'], 
            self.attention_encoder_config['expansion'], 
            vocab_size, 
            self.attention_encoder_config['vocab_max']
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

        ## shape: (seq_len, batch_size, embedding_size)
        encoder_out = self.encoder(embeddings, attention_encoder_mask)#.permute(1,0,2) 

        ## mean pooling layer
        encoder_out = self.mean_pool(encoder_out, attention_encoder_mask)

        return encoder_out

    def forward(self, sent1_batch, sent2_batch):
        """
        Performs the forward pass for each batch
        """
        ## batch size might change toward the end of epoch

        ## mask for input sentence padding indices
        sent1_mask = self.get_encoder_mask(sent1_batch)
        sent2_mask = self.get_encoder_mask(sent2_batch)

        ## Send each batch through the siamese network 
        sent_A = self.forward_once(sent1_batch, sent1_mask)
        self.sent_A_mh_att_matrix = self.encoder.layer_attention_matrices

        sent_B = self.forward_once(sent2_batch, sent2_mask)
        self.sent_B_mh_att_matrix = self.encoder.layer_attention_matrices

        ## Calculate embedding similarity
        ## cosine similarity used here, because it gives a better measurement
        similarity = cosine_similarity(sent_A, sent_B) 

        return similarity

    def get_encoder_mask(self, input_sentence_batch):
        src_mask = (input_sentence_batch != self.pad_index).unsqueeze(1).unsqueeze(2)
        ## src_mask shape: (batch_size, 1, 1, sen_len)

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

        ## calculate mean
        mean = summed_embeddings / summed_mask

        return mean



