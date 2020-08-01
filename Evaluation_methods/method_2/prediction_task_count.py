from __future__ import division
from __future__ import print_function
import sys
#sys.path.append("/root/pytorch")
from Evaluation_methods.method_2.attribute_prediction_count import Attribute_prediction

import torch
import torch.nn as nn
import numpy as np
from torch.autograd.variable import Variable
from torch.optim import Adam
from multi_categorical_gans.datasets.dataset import Dataset
import torch.nn.functional as F

PRINT_FORMAT = "epoch {:d}/{:d} : {:.05f}"


def categorical_variable_loss(prediction, batch_y):

    loss = F.mse_loss(prediction, batch_y)
    return loss

def pre_train(attribute_prediction,
              train_data_x,
              train_data_y,
              batch_size=100,
              ):

    train_loss = pre_train_epoch(attribute_prediction, train_data_x, train_data_y ,batch_size, optim=None)
    print(np.mean(train_loss))
    f = open("error_vae_y.txt", "a")
    f.writelines(str(np.mean(train_loss)) + '\n')
    f.close()
def pre_train_epoch(attribute_prediction, train_data_x, train_data_y, batch_size, optim=None):

    attribute_prediction.train(mode=False)
    losses = []
    for batch_x,batch_y in zip(train_data_x.batch_iterator(batch_size),train_data_y.batch_iterator(batch_size)):
        batch_x = Variable(torch.from_numpy(batch_x))
        batch_y = Variable(torch.from_numpy(batch_y))
        prediction = attribute_prediction(batch_x)
        loss = categorical_variable_loss(prediction, batch_y)
        losses.append(loss.data.numpy())
        del loss
    return losses

def main():
    features = np.load("/root/pytorch/data/pre-test.matrix", allow_pickle=True)
    features = np.load("/root/pytorch/data/pre-vae-test.matrix", allow_pickle=True)
    i = 14
    data = np.concatenate((features[:, 0:i], features[:, i+1:]), axis=1)

    print(features.shape)

    data_y = Dataset(features[:, i:i+1])
    data_x = Dataset(data)

    attribute_prediction = Attribute_prediction(data.shape[1])
    attribute_prediction.load_state_dict(torch.load("/root/pytorch/Evaluation_models/prediction_count.torch"))

    pre_train(attribute_prediction,
              data_x,
              data_y,
              )

if __name__ == "__main__":
    main()
