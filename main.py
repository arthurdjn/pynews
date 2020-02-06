# -*- coding: utf-8 -*-
# Created on Tue Jan 28 19:10:23 2020
# @author: arthurd


from pynews import NewsDataset, NewsModel, Trainer
from pynews.eval import eval_func, analyze_confusion_matrix

# from argparse import ArgumentParser

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import SGD, Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score, precision_score

# import matplotlib.pyplot as plt
# import seaborn as sns


if __name__ == "__main__":

    print("\n=============================================\n\t\tTRAIN\n=============================================\n")

    # 0/ Global Parameters 
    VOCAB_SIZE = 3000
    SPLIT = 0.9
    BATCH_SIZE = 32
    LEARNING_RATE = 0.09
    EPOCHS = 250
    
    print("===========================",
          "\nHyper Parameters",
          "\n---------------------------",
          "\nVocab Size :\t", VOCAB_SIZE,
          "\nSplit :\t\t", SPLIT,
          "\nBatch Size :\t", BATCH_SIZE,
          "\nLearning Rate :\t", LEARNING_RATE,
          "\nEpochs :\t", EPOCHS)
    
    # 1/ Get the dataset, and convert it in PyTorch dataset object
    path = "data/signal_20_obligatory1_train.tsv.gz"
    
    dataset = NewsDataset(path, vocab_size = VOCAB_SIZE)
    print("Classes : {0}".format(dataset.classes))

    # Save the text_vectorizer
    dataset.save_vecorizer()

    # 2/ Split the dataset into training and testing parts
    train_size = int(SPLIT * dataset.shape[0])
    dev_size = dataset.shape[0] - train_size
    
    train_dataset, test_dataset = random_split(dataset, [train_size, dev_size])
    
    
    # 3/ Load the dataset with PyTorch
    # See the docs to add more parameters
    train_loader = DataLoader(dataset    = train_dataset,
                              batch_size = BATCH_SIZE,
                              shuffle    = True) 
    
    
    # 4/ Train the dataset
    num_classes = len(dataset.classes)
    model = NewsModel(VOCAB_SIZE, 100, 100, 100, 100, num_classes)
    # model = nn.Sequential(nn.Linear(VOCAB_SIZE, 40), nn.ReLU(40, 80), nn.ReLU(80, 60), nn.Sigmoid(60, num_classes))
    
    #criterion = Adam([p for p in model.parameters() if p.requires_grad], lr = LEARNING_RATE)
    #criterion = BCEWithLogitsLoss()     # need to convert labels in float 
                                        # loss = criterion(outputs.squeeze(), labels.float())
    criterion = CrossEntropyLoss()     # need to convert labels in long
                                        # loss = criterion(outputs.squeeze(), labels.long())    (?)   
    optimizer = SGD(model.parameters(), lr = LEARNING_RATE, weight_decay=0.01)
    trainer = Trainer(model, train_loader)
    train_losses = trainer.run(criterion, optimizer, EPOCHS, LEARNING_RATE)

    # Get the loss
    print("2.1/ Loss for {0} epochs".format(EPOCHS))
    print(train_losses)

    print("\n=============================================\n\t\tTEST\n=============================================\n")
    # 5/ Test the model
    print("Load the model...")
    train_loader = DataLoader(dataset    = test_dataset,
                              batch_size = BATCH_SIZE,
                              shuffle    = True)
    print("Classes : {0}".format(dataset.classes))
    # Evaluate the model
    print("2.2/ Confusion Matrix")
    test_accuracy, test_predictions, test_labels, confusion_matrix = eval_func(train_loader, model)

    # Print the accuracy
    print("Accuracy {0}".format(test_accuracy))
    print("Per class accuracy :", confusion_matrix.diag() / confusion_matrix.sum(1))
    # Compute the TP, FT, ... for each classes
    print("Analyze the confusion matrix")
    analyze_confusion_matrix(confusion_matrix)

    # Print the precision / Recall
    print("\n2.3/ Precision & Recall")
    precision = precision_score(test_labels, test_predictions, average='macro')
    recall = recall_score(test_labels, test_predictions, average='macro')
    macro_f1 = f1_score(test_labels, test_predictions, average='macro')

    print("Precision = ", precision)
    print("Recall = ", recall)
    print("Macro-f1 =", macro_f1)


    # 6/ Save the model
    print("Save the model...")
    model_name = "model_test.pt"
    print("Model name : {0}".format(model_name))
    print(model_name)

    torch.save(model.state_dict(), model_name)
