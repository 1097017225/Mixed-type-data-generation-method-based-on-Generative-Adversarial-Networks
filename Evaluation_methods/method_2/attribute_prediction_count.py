from __future__ import print_function

import torch
import torch.nn as nn



class Attribute_prediction(nn.Module):
    def __init__(self,input_size):
        super(Attribute_prediction, self).__init__()

        previous_layer_size = input_size
        hidden_activation = nn.Tanh()
        hidden_sizes = [15]
        hidden_layers = []

        for layer_size in hidden_sizes:
            hidden_layers.append(nn.Linear(previous_layer_size,layer_size))
            hidden_layers.append(hidden_activation)
            previous_layer_size = layer_size

        self.hidden_layers = nn.Sequential(*hidden_layers)

        self.output_layer = nn.Linear(previous_layer_size, 1)
        self.output_activation = nn.Sigmoid()

    def forward(self, inputs):

        hidden = self.hidden_layers(inputs)
        hidden = self.output_layer(hidden)
        hidden = self.output_activation(hidden)
        return hidden