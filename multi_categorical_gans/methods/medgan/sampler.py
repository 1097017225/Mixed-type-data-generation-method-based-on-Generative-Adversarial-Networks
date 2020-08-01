from __future__ import print_function

import sys
sys.path.append("/root/pytorch")

import argparse
import torch

import numpy as np

from torch.autograd.variable import Variable

from multi_categorical_gans.methods.general.autoencoder import AutoEncoder
from multi_categorical_gans.methods.medgan.generator import Generator

from multi_categorical_gans.utils.categorical import load_variable_sizes_from_metadata
from multi_categorical_gans.utils.commandline import parse_int_list
from multi_categorical_gans.utils.cuda import to_cuda_if_available, to_cpu_if_available, load_without_cuda


def sample(autoencoder, generator, num_samples, num_features, count_feature, batch_size=100, code_size=128, temperature=None,
           round_features=False):

    #autoencoder, generator = to_cuda_if_available(autoencoder, generator)

    autoencoder.train(mode=False)
    generator.train(mode=False)

    samples = np.zeros((num_samples, num_features), dtype=np.float32)

    start = 0
    while start < num_samples:
        with torch.no_grad():
            noise = Variable(torch.FloatTensor(batch_size, code_size).normal_())
            #noise = to_cuda_if_available(noise)
            batch_code = generator(noise)
            batch_samples = autoencoder.decode(batch_code, training=False, temperature=temperature)
        #batch_samples = to_cpu_if_available(batch_samples)
        batch_samples = batch_samples.data.numpy()

        # if rounding is activated (for MedGAN with binary outputs)
        if round_features:
            batch_samples = np.round(batch_samples)

        #batch_samples[:,0:count_feature] = np.ceil(np.exp(batch_samples[:,0:count_feature]*12)-1)

        # do not go further than the desired number of samples
        end = min(start + batch_size, num_samples)
        # limit the samples taken from the batch based on what is missing
        samples[start:end, :] = batch_samples[:min(batch_size, end - start), :]

        # move to next batch
        start = end
    return samples


def main():
    options_parser = argparse.ArgumentParser(description="Sample data with MedGAN.")

    options_parser.add_argument("autoencoder", type=str, help="Autoencoder input file.")
    options_parser.add_argument("generator", type=str, help="Generator input file.")
    options_parser.add_argument("num_samples", type=int, help="Number of output samples.")

    options_parser.add_argument("data", type=str, help="Output data.")

    options_parser.add_argument("--metadata", type=str,
                                help="Information about the categorical variables in json format.")

    options_parser.add_argument(
        "--code_size",
        type=int,
        default=128,
        help="Dimension of the autoencoder latent space."
    )

    options_parser.add_argument(
        "--count_feature",
        type=int,
        default=15,
        help="Dimension of the count variable"
    )

    options_parser.add_argument(
        "--encoder_hidden_sizes",
        type=str,
        default="",
        help="Size of each hidden layer in the encoder separated by commas (no spaces)."
    )

    options_parser.add_argument(
        "--decoder_hidden_sizes",
        type=str,
        default="",
        help="Size of each hidden layer in the decoder separated by commas (no spaces)."
    )

    options_parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Amount of samples per batch."
    )

    options_parser.add_argument(
        "--generator_hidden_layers",
        type=int,
        default=2,
        help="Number of hidden layers in the generator."
    )

    options_parser.add_argument(
        "--generator_bn_decay",
        type=float,
        default=0.01,
        help="Generator batch normalization decay."
    )

    options_parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Gumbel-Softmax temperature."
    )

    options = options_parser.parse_args()

    if options.temperature is not None:
        temperature = options.temperature
    else:
        temperature = None

    variable_sizes = [2, 2, 2, 12, 2, 7, 29, 4, 3]
    autoencoder = AutoEncoder(
        sum(variable_sizes)+options.count_feature,
        code_size=options.code_size,
        encoder_hidden_sizes=parse_int_list(options.encoder_hidden_sizes),
        decoder_hidden_sizes=parse_int_list(options.decoder_hidden_sizes),
        variable_sizes=[options.count_feature]+variable_sizes
    )

    autoencoder.load_state_dict(torch.load(options.autoencoder))

    generator = Generator(
        code_size=options.code_size,
        num_hidden_layers=options.generator_hidden_layers,
        bn_decay=options.generator_bn_decay
    )
    generator.load_state_dict(torch.load(options.generator))
    #load_without_cuda(generator, options.generator)

    data = sample(
        autoencoder,
        generator,
        options.num_samples,
        sum(variable_sizes)+options.count_feature,
        options.count_feature,
        batch_size=options.batch_size,
        code_size=options.code_size,
        temperature=temperature,
        round_features=(temperature is None)
    )
    #data = data.astype(np.int)

    #data[:, 0:options.count_feature] = np.round(data[:, 0:options.count_feature], 4)
    #data[:, options.count_feature:] = data[:, options.count_feature:].astype(np.int)

    np.save(options.data, data)

if __name__ == "__main__":
    main()
