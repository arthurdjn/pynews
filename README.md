# pynews
[![GitHub Pipenv locked Python version](https://readthedocs.org/projects/pip/badge/)](http://pynews.readthedocs.io/en/latest/index.html)
[![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](http://pynews.readthedocs.io/en/latest/index.html)

**A PyTorch Multi Layer Perceptrons for NewsPaper classification.**

## Overview

### About

**PyNews** is PyTorch supervised document classifier. The model is a custom *Feed Forward Neural Network* and uses *Bag of Words* (BoW) 
from an unknown document as inputs. The outputs are possible sources of this document.

The model was trained on the **SAGA** server,  hosted at *Norwegian University of Science and Technology* (NTNU), in Norway.
The training dataset used is *The Signal Media One-Million News Articles*, which contains texts from news sites and 
blogs from September 2015.

### Dependencies

**PyNews** was made in Python 3.7 and uses multiple machine learning libraries :

  - PyTorch
  - Numpy
  - Pandas
  - Scikit-Learn
  - Pickle
  - Time

### Installation 

To use this package, clone the repository at https://github.uio.no/arthurd/pynews on your laptop and from the root folder, run on the commandline :

```
pip install .
```

This will install the package in your python environment and download all the latest dependencies. You can know use and tweak the parameters of pynews’ models.



## Running on SAGA

Furing this project, **SAGA** offered GPUs to train the model faster.

<p align="center">
  <img src="https://www.sigma2.no/themes/custom/sigma/logo.png">
  <br>
</p>

For our final model, the necessary items to run the testing data are stored on SAGA as follows:
- The trained vectorizers arthurd/in5550/assignment/pynews/vectorizer.pickle
- The final model is stored at arthurd/in5550/assignment/pynews/model.pt

Access is given on SAGA to these directories, and the function `eval_on_test.py` can be used to evaluate on the test data set.
We used a tokenizer to strip POS-tags from the documents, and specified this in the vectorizer we pickled. We have tested loading this pickled vectorizer, and it works as intended.

## Set Up

**PyNews** introduces basic steps for Natural Language Processing (NLP) analysis. First, you may want to clean up the raw data, from *The Signal Media One-Million News Articles* which contains articles from september 2015. This dataset contains Bag of Words (BoW) about 75.000 documents from 20 sources.

### Data Processing

The dataset can be found in the data folder, called *signal_20_obligatory1_train.tsv.gz.*

The dataset is made of Part of Speech (POS) tags for every words, meaning that the type (“NOUN”, “VERB”, “ADJ” etc.) are stacked on each words with an underscore. A first step was then to split the words in half, to keep only the word and not its type. Then, the BoW and the vocabulary can be created.

These functionalities are coded in the pynews.data module.

```python
import torch
from pynews import NewsDataset

PATH = "data/signal_20_obligatory1_train.tsv.gz"
# Limit of the vocabulary
VOCAB_SIZE = 3000

# Create the PyTorch Dataset object from the dataset
dataset = NewsDataset(PATH, vocab_size = VOCAB_SIZE)
```






