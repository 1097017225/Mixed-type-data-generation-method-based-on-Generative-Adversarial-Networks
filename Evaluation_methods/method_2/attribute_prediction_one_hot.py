from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attribute_prediction(nn.Module):
    def __init__(self,input_size):
        super(Attribute_prediction, self).__init__()

        previous_layer_size = input_size
        hidden_activation = nn.Tanh()
        hidden_sizes = [32]
        hidden_layers = []

        # self.input_norm = nn.BatchNorm1d(15, momentum=0.01)

        for layer_size in hidden_sizes:

            hidden_layers.append(nn.Linear(previous_layer_size,layer_size))
            #hidden_layers.append(nn.BatchNorm1d(layer_size, momentum=0.01))
            hidden_layers.append(hidden_activation)
            previous_layer_size = layer_size

        self.hidden_layers = nn.Sequential(*hidden_layers)

        self.output_layer = nn.Linear(previous_layer_size, 3)

    def forward(self, inputs, training=True, temperature=None):

        # input_count = self.input_norm(inputs[:,0:15])
        # input_one_hot = inputs[:,15:]
        #
        # inputs = torch.cat( (input_count, input_one_hot), dim=1 )

        hidden = self.hidden_layers(inputs)
        hidden = self.output_layer(hidden)
        return F.gumbel_softmax(hidden, hard=not training, tau=temperature)