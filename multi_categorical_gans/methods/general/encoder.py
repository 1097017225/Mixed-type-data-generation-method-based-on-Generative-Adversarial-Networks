from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable

class Encoder(nn.Module):

    def __init__(self, input_size, code_size, hidden_sizes=[]):
        super(Encoder, self).__init__()

        hidden_activation = nn.Tanh()

        previous_layer_size = input_size


        layer_sizes = list(hidden_sizes) + [code_size]
        layers = []

        for layer_size in layer_sizes:
            layers.append(nn.Linear(previous_layer_size, layer_size))
            #layers.append(nn.BatchNorm1d(layer_size, momentum=0.01))
            layers.append(hidden_activation)
            previous_layer_size = layer_size

        self.hidden_layers = nn.Sequential(*layers)

    def forward(self, inputs):

        return self.hidden_layers(inputs)
