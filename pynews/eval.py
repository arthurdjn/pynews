# -*- coding: utf-8 -*-
# Created on Wed Jan 29 14:45:57 2020
# @author: arthurd

import torch

def eval_func(batched_data, model):
    """
    Evaluate the model on the test data.

    Parameters
    ----------
    batched_data : torch DataLoader
        The loaded test data.
    model : torch Model
        The PyTorch model to evaluate.

    Returns
    -------
    accuracy : float
        Global accuracy of the model.
    predicted : torch tensor
        Predicted output.
    gold_label : torch tensor
        Truth output.
    confusion_matrix : torch tensor
        Confusion matrix.
    """
    
    # This function uses a model to compute predictions on data coming in batches.
    # Then it calculates the accuracy of predictions with respect to the gold labels.
    correct = 0
    total = 0
    predicted = None
    gold_label = None

    # Initialize the confusion matrix
    nb_classes = 20
    confusion_matrix = torch.zeros(nb_classes, nb_classes)

    with torch.no_grad():
        # Iterating over all batches (can be 1 batch as well):
        for n, (input_data, gold_label) in enumerate(batched_data):
            out = model(input_data)
            predicted = out.argmax(1)
            correct += len((predicted == gold_label).nonzero())
            total += len(gold_label)
            # Update the confusion matrix
            confusion_matrix[predicted, gold_label] += 1
        accuracy = correct / total

    return accuracy, predicted, gold_label, confusion_matrix


def analyze_confusion_matrix(confusion_matrix):
    """
    Analyse a confusion matrix by printing the True Positive (TP), False Positive (FP), ...
    and the specificity and sensitivity as well for each classes.

    Parameters
    ----------
    confusion_matrix : torch tensor of size (number_of classes, number_of_classes)
        The confusion matrix computed on the test data.

    Returns
    -------
    None.
    """
    
    n_classes = len(confusion_matrix)
    
    # True positive : correct prediction, ie the diagonal of the confusion matrix
    TP = confusion_matrix.diag()
    for c in range(n_classes):
        idx = torch.ones(n_classes).byte()
        idx[c] = 0
        TN = confusion_matrix[idx.nonzero()[:, None], idx.nonzero()].sum()
        FP = confusion_matrix[c, idx].sum()
        FN = confusion_matrix[idx, c].sum()

        sensitivity = (TP[c] / (TP[c] + FN))
        specificity = (TN / (TN + FP))
        
        # Display the analysis in the console
        print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(c, TP[c], TN, FP, FN))
        print("Sensitivity :", sensitivity)
        print("Specificity : {0}\n------".format(specificity))

