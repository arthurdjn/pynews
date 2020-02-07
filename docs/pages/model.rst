=====
Model
=====

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



Feature Tuning
==============

Before changing the structure of the model, we explored differents Bag of Words features implementation
by varying the vocabulary size, the preprocessing and building the vocabulary before and after the split.
For example, with a vocabulary size of 4000 we did not see improvement in the performance. In addition,
using the Part of Speech (POS) tags did not help to optimize the results. Because of difficulties and time
restriction we did not create the vocabulary after splitting into training and development parts.


Hyperparameters
===============

A brief training session to evaluate the performance with different hyper parameters was firstly performed.
The hyper parameters used are described on the table below.

+-------------------+-----------+
|Parameter          |Value      |
+-------------------+-----------+
|Split Train/Dev    | .9        |
+-------------------+-----------+
|Vocabulary         |3000       |
+-------------------+-----------+
|Batch size         | 32        |
+-------------------+-----------+
|Learning Rate      | .09       |
+-------------------+-----------+
|Epochs             | 250       |
+-------------------+-----------+





Architecture
============



Evaluation
==========



Model Training
==============

