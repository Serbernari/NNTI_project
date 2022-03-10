import pandas as pd
from preprocess import Preprocess
import logging
import torch
from torch.utils.data import DataLoader
from dataset import STSDataset
from datasets import load_dataset
import spacy
from torchtext.legacy.data import Field

logging.basicConfig(level=logging.INFO)

"""
For loading STS data loading and preprocessing
"""


class STSData:
    def __init__(
        self,
        dataset_name,
        columns_mapping,
        stopwords_path="stopwords-en.txt",
        model_name="lstm",
        max_sequence_len=512,
        normalization_const=5.0,
        normalize_labels=False,
    ):
        """
        Loads data into memory and create vocabulary from text field.
        """
        self.normalization_const = normalization_const
        self.normalize_labels = normalize_labels
        self.model_name = model_name
        self.max_sequence_len = max_sequence_len
        self.dataset_name = dataset_name
        ## load data file into memory
        self.load_data(dataset_name, columns_mapping, stopwords_path)
        self.columns_mapping = columns_mapping
        ## create vocabulary
        self.create_vocab()
        
        ## test to see if vocab works
        print('Testing vocab. Index for kids is {}'.format(self.vocab['kids']))

        s = self.dataset['train']['sentence_A'][0]
        s_t = self.vectorize_sequence(self.dataset['train']['sentence_A'][0])
        print('Testing vocab. Index for sentence {} is {}'.format(s, s_t))

        print(self.vocab.itos[0])
        print(self.vocab.itos[1])
        print(self.vocab.itos[2])

        ## print data columns
        print(self.dataset['train'].columns)

    def load_data(self, dataset_name, columns_mapping, stopwords_path):
        """
        Reads data set file from disk to memory using pandas
        """
        logging.info("loading and preprocessing data...")

        ## load datasets
        dataset = load_dataset(dataset_name)
        ## make pandas dataframes
        train = pd.DataFrame(dataset['train'])
        test = pd.DataFrame(dataset['test'])
        validation = pd.DataFrame(dataset['validation'])
        ## save to dictionary
        dataset = {'train':train, 'validation':validation, 'test':test}

        
        logging.info(dataset['train']['sentence_A'][42])
        logging.info(dataset['train']['sentence_B'][42])
        
        ## perform text preprocessing
        preprocessor = Preprocess(stopwords_path)
        self.dataset = preprocessor.perform_preprocessing(data=dataset, columns_mapping=columns_mapping)
        
        logging.info("reading and preprocessing data completed...")

        ## for comparison
        logging.info(self.dataset['train']['sentence_A'][42])
        logging.info(self.dataset['train']['sentence_B'][42])

    def create_vocab(self):
        """
        Creates vocabulary over entire text data field.
        """
        logging.info("creating vocabulary...")

        logging.info("loading spacy tokenizer")
        self.spacy_en = spacy.load('en_core_web_sm')

        logging.info("concatenating sentences")
        train_conc = self.dataset['train']['sentence_A'] + self.dataset['train']['sentence_B']
        logging.info("building vocab with torchtext")
        text_field = Field(
            tokenize=self.tokenize_src, 
            lower=True
        )

        text_field.build_vocab(train_conc, vectors='fasttext.simple.300d')
        self.vocab = text_field.vocab
        logging.info("creating vocabulary completed...")
    
    def tokenize_src(self, text):
        '''
        function to use when tokenizing
        '''
        return [tok.text for tok in self.spacy_en.tokenizer(text)]

    def data2tensors(self, data):
        """
        Converts raw data sequences into vectorized sequences as tensors
        """
        
        # vectorize sentences
        data['sentence_A']=data['sentence_A'].apply(self.vectorize_sequence)
        data['sentence_B']=data['sentence_B'].apply(self.vectorize_sequence)

        ## save sentence length (do we need this?)
        sent_A_lens = torch.tensor(data['sentence_A'].apply(len))
        sent_B_lens = torch.tensor(data['sentence_B'].apply(len))

        ## normalize label
        data['relatedness_score']=(data['relatedness_score']/self.normalization_const)

        ## pad sentences
        a_max_len = max(data['sentence_A'].apply(len))
        b_max_len = max(data['sentence_B'].apply(len))
        max_len = max(a_max_len, b_max_len)
        print('max len a: {}, max_len b: {}, max_len {}'.format(a_max_len, b_max_len, max_len))
        data['sentence_A'] = self.pad_sequences(data['sentence_A'], max_len)
        data['sentence_B'] = self.pad_sequences(data['sentence_B'], max_len)
        print(data['sentence_B'][69])

        ## dataframe to tensors
        sent_A_tensor = torch.tensor(data['sentence_A'])
        sent_B_tensor = torch.tensor(data['sentence_B'])
        target_tensor = torch.tensor(data['relatedness_score'])


        ## return STSDataSet object from data
        stsDatset = STSDataset(sent_A_tensor, sent_B_tensor, target_tensor,sent_A_lens, sent_B_lens, data['sentence_A'], data['sentence_B'])

        return stsDatset

    def get_data_loader(self, batch_size=8):
        ## turn data into tensors
        training_data = self.data2tensors(self.dataset['train'])
        test_data = self.data2tensors(self.dataset['test'])
        validation_data = self.data2tensors(self.dataset['validation'])

        ## one dataloader per split
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)

        return {'train':train_dataloader, 'test':test_dataloader, 'validation':validation_dataloader}

    def sort_batch(self, batch, targets, lengths):
        """
        Sorts the data, lengths and target tensors based on the lengths
        of the sequences from longest to shortest in batch
        """
        sents1_lengths, perm_idx = lengths.sort(0, descending=True)
        sequence_tensor = batch[perm_idx]
        target_tensor = targets[perm_idx]
        return sequence_tensor.transpose(0, 1), target_tensor, sents1_lengths

    def vectorize_sequence(self, sentence):
        """
        Replaces tokens with their indices in vocabulary
        """

        return [self.vocab.stoi[word.lower()] for word in sentence]

    def pad_sequences(self, vectorized_sents, sents_lengths):
        """
        Pads zeros at the end of each sequence in data tensor till max
        length of sequence in that batch
        """
        
        for sentence in vectorized_sents:
            while len(sentence) < sents_lengths:
                sentence.append(self.vocab.stoi['<pad>'])

        return vectorized_sents
