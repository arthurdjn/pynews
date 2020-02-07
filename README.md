# pynews
![GitHub Pipenv locked Python version](https://readthedocs.org/projects/pip/badge/)
[![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](http://shields.io/)

**A PyTorch Multi Layer Perceptrons for NewsPaper classification.**

## About

**PyNews** is PyTorch supervised document classifier. The model is a custom *Feed Forward Neural Network* and uses *Bag of Words* (BoW) 
from an unknown document as inputs. The outputs are possible sources of this document.

The model was trained on the **SAGA** server,  hosted at *Norwegian University of Science and Technology* (NTNU), in Norway.
The training dataset used is *The Signal Media One-Million News Articles*, which contains texts from news sites and 
blogs from September 2015.


## Running the test data set

For our final model, the necessary items to run the testing data are stored on SAGA as follows:
- The trained vectorizers arthurd/in5550/assignment/pynews/vectorizer.pickle
- The final model is stored at arthurd/in5550/assignment/pynews/model.pt

Access is given on SAGA to these directories, and the function `eval_on_test.py` can be used to evaluate on the test data set.
We used a tokenizer to strip POS-tags from the documents, and specified this in the vectorizer we pickled. We have tested loading this pickled vectorizer, and it works as intended.

