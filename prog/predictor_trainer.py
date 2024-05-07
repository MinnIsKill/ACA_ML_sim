"""
    File:     predictor_trainer.py
    Author:   Vojtěch Kališ
    VUTBR ID: xkalis03

    Brief:    PyTorch machine learning predictor trainer implementation
""" 

import torch.nn as nn

#future state predictor neural network model
class Future_state_predictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Future_state_predictor, self).__init__()
        #define the input layer, with the expected size of data input as well as
        #the size the hidden layer will expect as its input (so, input layer's output)
        self.fc1 = nn.Linear(input_size, hidden_size)
        #define the hidden layers with a ReLU activation function
        self.fc2 = nn.ReLU()
        #define the output layer, with the expected size of input as well as the size 
        #of the data output
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        #simply pass the data input first through the input layer,
        x = self.fc1(x)
        #then the hidden layer (applying the ReLU activation function),
        x = self.fc2(x)
        #and lastly the output layer
        x = self.fc3(x)

        return x
