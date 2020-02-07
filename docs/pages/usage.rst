=====
Usage
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


.. code-block:: python

    import torch
    from pynews import NewsDataset

    PATH = "data/signal_20_obligatory1_train.tsv.gz"
    # Limit of the vocabulary
    VOCAB_SIZE = 3000

    # Create the PyTorch Dataset object from the dataset
    dataset = NewsDataset(PATH, vocab_size = VOCAB_SIZE)


You can have access to the documents informations from the attributes of the NewsDataset class.

.. code-block:: pycon

    # The shape of the dataset 
    dataset.shape
    >>> (75141, 3000)

    # The classes of the dataset
    dataset.classes
    >>>    ['4 Traders', 'App.ViralNewsChart.com', 'BioSpace', 'Bloomberg', 'EIN News',
            'Fat Pitch Financials', 'Financial Content', 'Individual.com',
            'Latest Nigerian News.com', 'Mail Online UK', 'Market Pulse Navigator',
            'Marketplace', 'MyInforms', 'NewsR.in', 'Reuters', 'Town Hall' 'Uncova',
            'Wall Street Business Network', 'Yahoo! Finance', 'Yahoo! News Australia']

    # The BoW inputs
    dataset.input_features
    >>> tensor([[0., 0., 0.,  ..., 0., 0., 0.],
                [0., 0., 0.,  ..., 0., 0., 0.],
                [0., 0., 0.,  ..., 0., 0., 0.],
                ...,
                [0., 0., 0.,  ..., 0., 0., 0.],
                [0., 0., 0.,  ..., 0., 0., 0.],
                [0., 0., 0.,  ..., 0., 0., 0.]])
    
    # The gold classes 
    dataset.gold_classes
    >>> tensor([16,  0, 12,  ..., 19, 11, 12], dtype=torch.int32)


You can now save the vectorizer so your Bag of Words features will always be in the same order.

.. code-block:: python

    dataset.save_vectorizer("vectorizer.pickle")


It is recommended to split the dataset in two parts :
one used to train the model, the other to evaluate and test it.


.. code-block:: python

    # Define your split ratio
    SPLIT = 0.9

    train_size = int(SPLIT * dataset.shape[0])
    dev_size = dataset.shape[0] - train_size
    
    train_dataset, test_dataset = random_split(dataset, [train_size, dev_size])



Training
========

Before training the model, divide your dataset in batches and load it with the PyTorch class :

.. code-block:: python

    # Divide your data in batches of size BATCH_SIZE
    BATCH_SIZE = 32

    train_loader = DataLoader(dataset    = train_dataset,
                              batch_size = BATCH_SIZE,
                              shuffle    = True) 


Then, create your model or use the *NewsModel* one, and define your loss function and optimizer.

.. code-block:: python

    from pynews import NewsModel

    # Define the hyperparameters
    EPOCHS = 250
    LEARNING_RATE = 0.09
    WEIGHT_DECACAY = 0.01

    # Create a Feed Forward neural network
    # with 3 hidden layers
    # of 150 neurons each
    num_classes = len(dataset.classes)
    model = NewsModel(VOCAB_SIZE, 150, 150, 150, num_classes)

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    # Optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECACAY)


You can now train the model with :

.. code-block:: python

    from pynews import Trainer

    # Create your trainer for your model
    trainer = Trainer(model, train_loader)

    # Run it with the hyper parameters you defined
    train_losses = trainer.run(criterion, optimizer, EPOCHS, LEARNING_RATE)



Evaluate
========

Now that your model is trained, evaluate it on the test dataset.


.. code-block:: pycon

    # Load the dataset
    train_loader = DataLoader(dataset    = test_dataset,
                              batch_size = BATCH_SIZE,
                              shuffle    = True)

    # Evaluate the model
    test_accuracy, test_predictions, test_labels, confusion_matrix = eval_func(train_loader, model)


.. code-block:: console

    # Get the per class test_accuracy
    confusion_matrix.diag() / confusion_matrix.sum(1) 
    >>> 