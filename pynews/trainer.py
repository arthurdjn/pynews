# -*- coding: utf-8 -*-
# Created on Sun Jan 26 17:15:42 2020
# @author: arthurd


import time

from torch.optim import SGD
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable

# from torch.utils.tensorboard import SummaryWriter



class Trainer:
    """
    Train a model with a dataset.
    """
    
    def __init__(self, model, train_loader, model_name="model"):
        """
        Innitialize the trainer with the model and the dataset.

        Parameters
        ----------
        model : torch.nn.Module
            Custom model used.
        train_loader : torch.utils.data.DataLoader
            Data loaded for PyTorch. This data will be used to train the model.
        train_losses : list
            The loss during the training.
        """
        
        self.model = model
        self.train_loader = train_loader
        self.model_name = model_name
        
    def run(self, criterion, optimizer, epochs, lr):
        """
        Method to run the trainer.
        Use this function to train your model with your data.

        Parameters
        ----------
        criterion : torch.nn.modules.loss
            The loss function.
        optimizer : torch.optim
            Method to optimize the model.
        epochs : int
            Number of iterations.
        lr : float
            Learning rate.

        Returns
        -------
        train_losses : list
            List of size epochs, containing
        """
        
        # Print the details     
        print("---------------------------",
              "\nActivation :\t", self.model.activation,
              "\nLoss :\t\t", criterion,
              "\nOptimizer : \t", optimizer.__class__,
              "\n===========================\n")

        train_losses = []
        # Loop over the dataset epochs times
        for epoch in range(epochs):
            # Loss of the training
            epoch_losses = 0.0
            # Loop over the dataset train_loader batches
            start = time.time()
            for (idx, data) in enumerate(self.train_loader, 0):               
                
                # 1/ Get the data from the batch
                # Get the inputs and labels from the dataset
                inputs, labels = data
                            
                # 2/ Compute the forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels.long())       
                self.model.zero_grad()
                
                # 3/ Compute the backward pass
                # Perform the backward pass
                loss.backward()
                # Update the weights
                optimizer.step()
                
                # 4/ Get the stats for each epochs
                # Print statistics
                epoch_losses += loss.item()
                
                # Display the loss every 100 batches
                # if idx % 100 == 0:    # print every 2000 mini-batches
                #     print('[%d, %5d ]\tloss: %.3f' %
                #           (epoch, idx, epoch_losses / ((idx + 1) * self.train_loader.batch_size)))
                #     print("Batch n°{0} in {1:.3f}s".format(idx, time.time() - start))
                #     start = time.time()
                    
            train_losses.append(epoch_losses/len(self.train_loader))
            
            # Display the total training duration
            print("Training duration : {0}".format(time.time() - start))
        
        return train_losses
                
                
                