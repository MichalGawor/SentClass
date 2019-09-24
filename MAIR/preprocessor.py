import csv
from bert_serving.client import BertClient
from keras.utils.np_utils import to_categorical
import os
import pandas as pd
import sys

from utils import Vocabulary


class BertPreprocessor():
    '''Bert service client class.'''
    def __init__(self, bert_model='bert-base-cased', vector_length=1024):
        #TODO try to run BERT service with API as subprocess
        #see https://github.com/hanxiao/bert-as-service
        self.vector_lenght = vector_length
        self.bert = BertClient()
        self._label_vocab = Vocabulary(lower=False, unk_token=False)

    def __len__(self):
        return self.vector_length

    def build(self, labels):
        '''Build preprocessor vocabulary

        Args:
            labels: List-like of labels
        
        Returns:
            self
        '''
        self._label_vocab.add_documents(labels)
        self._label_vocab.build()
        return self

    def transform(self, sentences, labels=None):
        '''Return sentence embeddings and labels in numerical form

        Args:
            sentences: List-like of sentences
            <optional> labels: List-like of labels; None by default

        Returns:
            embeddings (ndarray): array of size n-sentences x m-bert vector length
            <optional> y ([]): list of corresponding labels ID, returned if labels are provided
        '''


        vector_vocab = self.bert.encode(sentences)
        if labels is not None:
            y = [self._label_vocab.token_to_id(label) for label in labels]
            y = to_categorical(y, self.label_size).astype(int)
            return vector_vocab, y
        else:
            return vector_vocab


    def inverse_transform(self, y):
        '''Return labels from labels ID

        Args:
            y: List-like of label IDs

        Returns:
            labels: list of labels
        '''

        inverse_y = [self._label_vocab.id2doc(ids) for ids in y]
        return inverse_y

    @property
    def __len__(self):
        #TODO set it with running bert as subprocess
        return self.vector_length


    @property
    def label_size(self):
        return len(self._label_vocab.vocab)
