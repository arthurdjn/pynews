# -*- coding: utf-8 -*-
# Created on Sun Jan 26 17:15:42 2020
# @author: arthurd



from torch.nn import Module, Linear, Sigmoid, ReLU, Tanh, ModuleList, Softmax
import torch.nn.functional as F


class NewsModel(Module):
    """
    Newspaper Model.
    """
    
    def __init__(self, *layers_size):
        """
        Initialize the model with the number of neurons in each hidden layers.
        The ReLU function was used as activation.
        
        Parameters
        ----------
        *layers_size : int
            Number of neurons in the hidden layers.

        Returns
        -------
        None.
        """
        
        super(NewsModel, self).__init__()
        
        # Save the layer's size
        self.layers_size = [size for size in layers_size]
        # Creating layers
        self.layers = ModuleList([Linear(layers_size[i-1], layers_size[i]) for i in range(1, len(layers_size))])
        
        # Activation function used for the model
        self.activation = ReLU()
        
        
    def forward(self, inputs):
        """
        Predict the outputs from the given inputs.
        The ReLU function is used from the inputs to the last hidden layer,
        then a linear function from the last hidden layer to the outputs.
        
        Parameters
        ----------
        inputs : torch tensor
            Inputs tensor.

        Returns
        -------
        outputs : torch tensor
            Predicted tensor.
        """
        
        # Step 1
        # Change this if you want another activation function from the input layer
        # to the first hidden layer
        outputs = F.relu(self.layers[0](inputs))
        
        # Step 2
        # The same activation function is used from one hidden layer to the other
        for layer in self.layers[1: -1]:
            outputs = self.activation(layer(outputs))
        
        # Step 3
        # Change this if you want another activation function from the last hidden layer
        # to the output layer
        outputs = self.layers[-1](outputs) 
        
        # Final prediction
        return outputs
        