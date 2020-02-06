# -*- coding: utf-8 -*-
# Created on Sun Jan 26 17:15:42 2020
# @author: arthurd


import numpy as np
import pandas as pd
import pickle

import torch
from torch.utils.data import Dataset

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder



class NewsDataset(Dataset):
    """
    NewsDataset class of different newspaper.  
    The class heritates from torch.utils.data.DataSet class.    
    """
    
    def __init__(self, path, vocab_size=3000):
        """
        Initialize the DataSet from a Bag of Words database.
        Works with full text documents, but it is not optimum.

        Parameters
        ----------
        path : str
            Path to the database file. The database should contains two columns :
                - 'source'
                - 'text'
            The first column is the source of the newspaper/article.
            The second column is its content.
        vocab_size : int, optional
            Length of the vocabulary. It is the maximum length of the vocabulary,
            meaning that each inputs have a max length of vocab_size.
            The default is 3000.
        """
        
        
        # Read the dataset
        train_dataset = pd.read_csv(path, sep='\t', header=0, compression='gzip')
        classes = train_dataset['source']  # Array with correct classes
        texts = train_dataset['text']
        
        # Class Vectorizer
        # CountVectorizer creates a bag-of-words vector, using at most 'max_features' words
        text_vectorizer = CountVectorizer(max_features  = vocab_size,     # Size of the vocabulary
                                          tokenizer     = self.tokenizer, # Tokenize the texts, to keep only words and not the types
                                          stop_words    = "english",      # Do not take into account the stop words
                                          strip_accents = 'unicode', 
                                          lowercase     = False, 
                                          binary        = True)           # True indicates binary BoW
        
        self.text_vectorizer = text_vectorizer

        # LabelEncoder returns integers for each label
        label_vectorizer = LabelEncoder()
        
        # Create the BoW from the Vectorizer object
        # We specify float32 because the default, float64, is inappropriate for pytorch:
        input_features = text_vectorizer.fit_transform(texts.values).toarray().astype(np.float32)
        gold_classes = label_vectorizer.fit_transform(classes.values)
        
        # Number of classes:
        classes = label_vectorizer.classes_
        self.classes = classes
        
        # Create a PyTorch object for the data
        # Convert the array to tensor
        torch_input_features = torch.from_numpy(input_features)
        torch_gold_classes = torch.from_numpy(gold_classes)
        self.input_features = torch_input_features
        self.gold_classes = torch_gold_classes
        
        # Shape of the dataset : Training_size * Vocab_size
        self.shape = (len(self.input_features), len(self.input_features[0]))
              
        
    def __getitem__(self, index):
        return (self.input_features[index, :], self.gold_classes[index])
    
    def __len__(self):
        return self.shape[0]
    
    def save_vecorizer(self, filename = "vectorizer.pickle"):
        """
        Save the text_vectorizer method used to transform the dataset into Bag of Words tensors.
        This file is saved with a binary pickle format.

        Parameters
        ----------
        filename : str, optional
            Name of the text vectorizer file. 
            The default is "vectorizer.pickle".

        Returns
        -------
        None.
        """
        
        with open(filename, 'wb') as f:
            pickle.dump(self.text_vectorizer, f)
    
    
    def tokenizer(self, sample):
        """
        Tokenize a text and extract only the words.
        
        Parameters
        ----------
        sample : list
            Text sample to tokenize.

        Returns
        -------
        sample : list
            Tokenized text, containing only words.
        """
        
        # Split the words in half, and keep only the first index.
        # The first index corresponds to the word, the last to the type (eg : VERB, NUM...)
        sample = [w.rsplit("_", 1)[0] for w in sample.split()]
        return sample
    
