from __future__ import division
from __future__ import print_function
import sys
sys.path.append("/root/pytorch")
from Evaluation_methods.method_2.attribute_prediction_one_hot import Attribute_prediction

import torch
import torch.nn as nn
import numpy as np
from torch.autograd.variable import Variable
from torch.optim import Adam
from multi_categorical_gans.datasets.dataset import Dataset
import torch.nn.functional as F

PRINT_FORMAT = "epoch {:d}/{:d} : {:.05f}"


def categorical_variable_loss(prediction, batch_y):
    target = torch.argmax(batch_y, dim=1)
    loss = F.cross_entropy(prediction, target)
    return loss

def pre_train(attribute_prediction,
              train_data_x,
              train_data_y,
              output_path,
              batch_size=100,
              start_epoch=0,
              num_epochs=100,
              l2_regularization=0,
              learning_rate=0.002,
              temperature=0.666
              ):
    optim = Adam(attribute_prediction.parameters(), weight_decay=l2_regularization, lr=learning_rate)
    for epoch_index in range(start_epoch, num_epochs):
        train_loss = pre_train_epoch(attribute_prediction, train_data_x, train_data_y ,batch_size, optim, temperature)
        torch.save(attribute_prediction.state_dict(), output_path)
        print(PRINT_FORMAT.format(epoch_index+1,num_epochs,np.mean(train_loss)))



def pre_train_epoch(attribute_prediction, train_data_x, train_data_y, batch_size, optim=None, temperature=None):
    #autoencoder.train(mode=(optim is not None))

    training = optim is not None
    attribute_prediction.train(mode=True)
    losses = []
    for batch_x,batch_y in zip(train_data_x.batch_iterator(batch_size),train_data_y.batch_iterator(batch_size)):
        if optim is not None:
            optim.zero_grad()

        batch_x = Variable(torch.from_numpy(batch_x))
        batch_y = Variable(torch.from_numpy(batch_y))

        prediction = attribute_prediction(batch_x, training=True, temperature=temperature)

        loss = categorical_variable_loss(prediction, batch_y)
        loss.backward()

        if training:
            optim.step()

        losses.append(loss.data.numpy())
        del loss
    return losses

def initialize_weights(module):
    if type(module) == nn.Linear:
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif type(module) == nn.BatchNorm1d:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)

def main():
    features = np.load("/root/pytorch/data/pre-train.matrix", allow_pickle=True)

    i= 75
    j= 78

    data = np.concatenate((features[:, 0:i], features[:, j:]), axis=1)

    data_y = Dataset(features[:,i:j])
    data_x = Dataset(data)

    attribute_prediction = Attribute_prediction(data.shape[1])
    attribute_prediction.apply(initialize_weights)
    print(features.shape)
    pre_train(attribute_prediction,
              data_x,
              data_y,
              output_path="/root/pytorch/Evaluation_models/prediction_one_hot.torch"
              )

if __name__ == "__main__":
    main()
