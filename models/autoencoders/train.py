#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from models.autoencoders.model import variational_simple, direct_simple, variational, direct, l1norm, corr
from models.autoencoders.model import vae_conv2lstm_1D, vae_conv2lstm
from models.autoencoders.model import vae_d2lstm, vae_d2lstm_1D

def main():

    config = {
        'batch_size': 512,
        'holdout_size': 1000,
        'n_epochs': 5,
        'latent_dim': 8,
        'inner_dim': 80,
        'output_dim': 10,
        'n_layers': 1,
        'n_dense': 3,
        'beta': 0.5,
        'learning_rate_decay': 0.000,
        'learning_rate': 0.001,
        'activation': 'relu',
        'dropout': 0.25,
        'weight_initialization': 'he_normal',
        'reduce_lr': False,
        'obs_per_timestep': 10,
        'n_features': 50,
        'conv_filter_n': 64
    }

    data = np.random.random_sample((2048, config['n_features'] // config['obs_per_timestep'], config['obs_per_timestep']))

    print('Checking vae_conv2lstm_1D...')
    X = data.reshape((-1, config['n_features']))
    model, encoder, latent_statistics, sampler, _ = vae_conv2lstm_1D(config)
    model.fit(X, X, batch_size=config['batch_size'], epochs=config['n_epochs'], shuffle=True, verbose=0)

    print('Checking vae_conv2lstm...')
    X = data
    model, encoder, latent_statistics, sampler, _ = vae_conv2lstm(config)
    model.fit(X, X, batch_size=config['batch_size'], epochs=config['n_epochs'], shuffle=True, verbose=0)

    print('Checking vae_d2lstm_1D...')
    X = data.reshape((-1, config['n_features']))
    model, encoder, latent_statistics, sampler, _ = vae_d2lstm_1D(config)
    model.fit(X, X, batch_size=config['batch_size'], epochs=config['n_epochs'], shuffle=True, verbose=0)

    print('Checking vae_d2lstm...')
    X = data
    model, encoder, latent_statistics, sampler, _ = vae_d2lstm(config)
    model.fit(X, X, batch_size=config['batch_size'], epochs=config['n_epochs'], shuffle=True, verbose=0)

    print('Checking variational_simple...')
    X = data.reshape((-1, config['n_features']))
    model, encoder, latent_statistics, sampler, _ = variational_simple(config)
    model.fit(X, X, batch_size=config['batch_size'], epochs=config['n_epochs'], shuffle=True, verbose=0)

    print('Checking direct_simple...')
    X = data.reshape((-1, config['n_features']))
    model, encoder, latent_statistics, sampler, _ = direct_simple(config)
    model.fit(X, X, batch_size=config['batch_size'], epochs=config['n_epochs'], shuffle=True, verbose=0)

    print('Checking variational...')
    X = data
    model, encoder, latent_statistics, sampler, _ = variational(config)
    model.fit(X, X, batch_size=config['batch_size'], epochs=config['n_epochs'], shuffle=True, verbose=0)

    print('Checking direct...')
    X = data
    model, encoder, end_to_end = direct(config)
    model.fit(X, X, batch_size=config['batch_size'], epochs=config['n_epochs'], shuffle=True, verbose=0)

    print('Checking l1norm...')
    X = data
    model, encoder, end_to_end = l1norm(config)
    model.fit(X, X, batch_size=config['batch_size'], epochs=config['n_epochs'], shuffle=True, verbose=0)

    print('Checking corr...')
    X = data
    model, encoder, end_to_end = corr(config)
    model.fit(X, X, batch_size=config['batch_size'], epochs=config['n_epochs'], shuffle=True, verbose=0)

if __name__ == "__main__":
    main()