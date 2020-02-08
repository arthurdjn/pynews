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

For this project **SAGA** offered GPUs to train the model faster.

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

You can have access to the documents informations from NewsDataset attributes.

```python
# The shape of the dataset
>>> dataset.shape
(75141, 3000)

# The classes of the dataset
>>> dataset.classes
['4 Traders', 'App.ViralNewsChart.com', 'BioSpace', 'Bloomberg', 'EIN News',
 'Fat Pitch Financials', 'Financial Content', 'Individual.com',
 'Latest Nigerian News.com', 'Mail Online UK', 'Market Pulse Navigator',
 'Marketplace', 'MyInforms', 'NewsR.in', 'Reuters', 'Town Hall' 'Uncova',
 'Wall Street Business Network', 'Yahoo! Finance', 'Yahoo! News Australia']

# The BoW inputs
>>> dataset.input_features
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
         ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]])

# The gold classes
>>> dataset.gold_classes
tensor([16,  0, 12,  ..., 19, 11, 12], dtype=torch.int32)
```

You can now save the vectorizer so your Bag of Words features will always be in the same order.

```python
dataset.save_vectorizer("vectorizer.pickle")
```

It is recommended to split the dataset in two parts : one used to train the model, the other to evaluate and test it.

```python
# Define your split ratio
SPLIT = 0.9

train_size = int(SPLIT * dataset.shape[0])
dev_size = dataset.shape[0] - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, dev_size])
```

### Training

Before training the model, divide your dataset in batches and load it with the PyTorch class :

```python
# Divide your data in batches of size BATCH_SIZE
BATCH_SIZE = 32

train_loader = DataLoader(dataset    = train_dataset,
                          batch_size = BATCH_SIZE,
                          shuffle    = True)
```

Then, create your model or use the NewsModel one, and define your loss function and optimizer.

```python
from pynews import NewsModel

# Define the hyperparameters
EPOCHS = 250
LEARNING_RATE = 0.09
WEIGHT_DECAY = 0.01

# Create a Feed Forward neural network
# with 3 hidden layers
# of 150 neurons each
num_classes = len(dataset.classes)
model = NewsModel(VOCAB_SIZE, 150, 150, 150, num_classes)

# Loss function
criterion = torch.nn.CrossEntropyLoss()
Optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
```

You can now train the model with :

```python
from pynews import Trainer

# Create your trainer for your model
trainer = Trainer(model, train_loader)

# Run it with the hyper parameters you defined
train_losses = trainer.run(criterion, optimizer, EPOCHS, LEARNING_RATE)
```

### Testing

Now that your model is trained, evaluate it on the test dataset.

```python
# Load the dataset
train_loader = DataLoader(dataset    = test_dataset,
                          batch_size = BATCH_SIZE,
                          shuffle    = True)

# Evaluate the model
test_accuracy, test_predictions, test_labels, confusion_matrix = eval_func(train_loader, model)
# Get the per class accuracy
per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
# Compute the precision, recall and macro-f1 scores
precision = precision_score(test_labels, test_predictions, average='macro')
recall = recall_score(test_labels, test_predictions, average='macro')
macro_f1 = f1_score(test_labels, test_predictions, average='macro')
```

```python
# Global accuracy
>>> test_accuracy
0.5281437125748503

# Per class accuracy
>>> per_class_accuracy
tensor([0.4314, 0.8500, 0.1333, 0.2852, 0.8547, 0.2279, 0.1297, 0.5329, 0.5388,
        0.5556, 0.1435, 0.2082, 0.3446, 0.7043, 0.5000, 0.1399, 0.4604, 0.1401,
        0.3069, 0.4850])

# Precision, recall and macro-F1 scores
>>> precision
0.4126984126984127
>>> recall
0.4944444444444444
>>> macro_f1
0.4358730158730159
```

## Example

In this example, we will try to predict the source of an unknown document.

Open your model and the vectorizer.pickle to process the data exactly the same way as the training data.

```python
import numpy as np

import torch
from torch.utils import data

# Load your pytorch model
model = torch.load("your_model_path.pt")
# Open your vectorizer file
with open("vectorizer.pickle", 'rb') as f:
        text_vectorizer = pickle.load(f)  # Loading the vectorizer
```

Download the article from one of the 20 classes used to train the model. For example, we will try to predict the source of this article (https://myinforms.com/banking-real-estate/best-bank-in-canada/).

```python
document = """
            Best Bank in Canada

            There was a time when you had to physically haul yourself...
            #....
            """
```

Extract the Bag of Words features and the vocabulary of the document with the vectorizer.

```python
# The Bag of Word
input_features = text_vectorizer.transform([document]).toarray().astype(np.float32)
# Converting the numpy array to pytorch tensors
torch_input_features = torch.from_numpy(input_features)
```

At this point, you can now predict the source.

```python
# Run the model on the unknown document
prediction = model(torch_input_features)
predicted = prediction.argmax(1)
# Classes generated by the training processing
classes = ['4 Traders', 'App.ViralNewsChart.com', 'BioSpace', 'Bloomberg', 'EIN News',
           'Fat Pitch Financials', 'Financial Content', 'Individual.com',
           'Latest Nigerian News.com', 'Mail Online UK', 'Market Pulse Navigator',
           'Marketplace', 'MyInforms', 'NewsR.in', 'Reuters', 'Town Hall' 'Uncova',
           'Wall Street Business Network', 'Yahoo! Finance', 'Yahoo! News Australia']
# Get the name of the predicted class
predicted_class = classes[predicted]
```

```python
>>> predicted_class
MyInforms
```

That's it !










