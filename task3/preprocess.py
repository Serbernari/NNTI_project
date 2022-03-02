import re
import string 

"""
Performs basic text cleansing on the unstructured field 
"""


class Preprocess:
    def __init__(self, stpwds_file_path):
        """
        Initializes regex patterns and loads stopwords
        """
        # TODO implement
        self.stop_words = []

        with open(stpwds_file_path, "r") as f:
            self.stop_words = f.read().split()
        #rint(self.stop_words)

        self.punctuation_regex = '[{}]'.format(string.punctuation)
        

    def perform_preprocessing(self, data, columns_mapping):
        ## TODO normalize text to lower case
        data = self.lower_case(data)

        ## TODO remove punctuations
        data = self.remove_punctuation(data)

        ## TODO remove stopwords
        data = self.remove_stopwords(data)
        ## TODO add any other preprocessing method (if necessary)
        return data

    def lower_case(self, dataset):
        dataset['train']['sentence_A'] = dataset['train']['sentence_A'].str.lower()
        dataset['train']['sentence_B'] = dataset['train']['sentence_B'].str.lower()
        dataset['test']['sentence_A'] = dataset['test']['sentence_A'].str.lower()
        dataset['test']['sentence_B'] = dataset['test']['sentence_B'].str.lower()
        dataset['validation']['sentence_A'] = dataset['validation']['sentence_A'].str.lower()
        dataset['validation']['sentence_B'] = dataset['validation']['sentence_B'].str.lower()

        return dataset


    def remove_punctuation(self, dataset):
        dataset['train']['sentence_A'] = dataset['train']['sentence_A'].str.replace(self.punctuation_regex, '')
        dataset['train']['sentence_B'] = dataset['train']['sentence_B'].str.replace(self.punctuation_regex, '')
        dataset['test']['sentence_A'] = dataset['test']['sentence_A'].str.replace(self.punctuation_regex, '')
        dataset['test']['sentence_B'] = dataset['test']['sentence_B'].str.replace(self.punctuation_regex, '')
        dataset['validation']['sentence_A'] = dataset['validation']['sentence_A'].str.replace(self.punctuation_regex, '')
        dataset['validation']['sentence_B'] = dataset['validation']['sentence_B'].str.replace(self.punctuation_regex, '')

        return dataset

    def remove_stopwords(self, dataset):
        dataset['train']['sentence_A'] = dataset['train']['sentence_A'].apply(lambda sentence: [word for word in sentence.split() if word not in self.stop_words])
        dataset['train']['sentence_B'] = dataset['train']['sentence_B'].apply(lambda sentence: [word for word in sentence.split() if word not in self.stop_words])
        dataset['test']['sentence_A'] = dataset['test']['sentence_A'].apply(lambda sentence: [word for word in sentence.split() if word not in self.stop_words])
        dataset['test']['sentence_B'] = dataset['test']['sentence_B'].apply(lambda sentence: [word for word in sentence.split() if word not in self.stop_words])
        dataset['validation']['sentence_A'] = dataset['validation']['sentence_A'].apply(lambda sentence: [word for word in sentence.split() if word not in self.stop_words])
        dataset['validation']['sentence_B'] = dataset['validation']['sentence_B'].apply(lambda sentence: [word for word in sentence.split() if word not in self.stop_words])
        return dataset