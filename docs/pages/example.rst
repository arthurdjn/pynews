=======
Example
=======

PyNews introduces basic steps for Natural Language Processing (NLP) analysis.
First, you may want to clean up the raw data, from *The Signal
Media One-Million News Articles* which contains articles from september 2015. This dataset contains Bag of Words (BoW) about 75.000 documents from 20 sources. 


Data Processing
===============

The dataset can be found in the *data* folder, called *signal_20_obligatory1_train.tsv.gz*.

The dataset is made of Part of Speech (POS) tags for every words, meaning that the type ("NOUN", "VERB", "ADJ" etc.) are stacked on each words with an underscore.
A first step was then to split the words in half, to keep only the word and not its type.
Then, the BoW and the vocabulary can be created.


These functionalities are coded in the *pynews.data* module. 


.. code-block:: python

    import torch
    from pynews import NewsDataset

    PATH = "data/signal_20_obligatory1_train.tsv.gz"
    # Limit of the vocabulary
    VOCAB_SIZE = 3000

    # Create the PyTorch Dataset object from the dataset
    dataset = NewsDataset(PATH, vocab_size = VOCAB_SIZE)




Hyper Parameters
================



Feature Tuning
==============


Architecture
============



Evaluation
==========



Model Training
==============

