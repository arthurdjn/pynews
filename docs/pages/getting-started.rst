===============
Getting Started
===============


About
=====

This project concerns news article classification using Bag of Words (BoW). A feed forward neural network was
trained to predict one of the 20 sources. The **PyNews** package implemented in python was used to explore the
different possibilities and training hyper parameters, with the help of PyTorch.
We have trained and compared five different architectures on a training and development set. The main focus
of this report is to analyse the influence of the number of hidden layers on the model’s performance and time
efficiency.
From these different structures, we selected the best one and trained it three times and will be used to predict
unseen data.


Dependencies
============

**PyNews** was made in Python 3.7 and uses multiple machine learning libraries :

- PyTorch
- Numpy
- Pandas
- Scikit-Learn
- Pickle
- Time


Instalation
===========

To use this package, clone the repository at https://github.uio.no/arthurd/pynews on your laptop and from the root folder, 
run on the commandline :

``pip install .``

This will install the package in your python environment and download all the latest dependencies. You can know use and tweak the parameters of pynews' models.




