#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from extractor.autoencoder.autoencoder import Autoencoder
from extractor.autoencoder.losses import total_loss

class Corr(Autoencoder):

    def __init__(self):
        super().__init__()
        self.name = 'corr'

    def build(self, features=0, obs_per_timestep=0, settings=None):
        super().build(features, obs_per_timestep, settings)
        n_timesteps = int(features / obs_per_timestep)
        shape = [None, n_timesteps, obs_per_timestep]
        rnn = tf.keras.layers.LSTM

        # Encoder
        x = tf.keras.layers.Input(shape=(shape[1], shape[2]))
        h = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(settings['inner_dim']), input_shape=(shape[1], shape[2]))(x)
        for i in range(settings['layers']):
            h = rnn(settings['inner_dim'], return_sequences=True)(h)
        encoded = rnn(settings['inner_dim'], input_shape=(None, settings['inner_dim']))(h)
        z = tf.keras.layers.Dense(settings['latent_dim'])(encoded)

        # Decoder
        z_repeated = tf.keras.layers.RepeatVector(shape[1])(z)
        for i in range(settings['layers']):
            z_repeated = rnn(settings['inner_dim'], return_sequences=True)(z_repeated)
        decoded = rnn(settings['inner_dim'], return_sequences=True)(z_repeated)
        x_estimate = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(settings['output_dim']))(decoded)

        self.encoder = tf.keras.models.Model(inputs=[x], outputs=[z])
        self.model = tf.keras.models.Model(inputs=[x], outputs=[x_estimate])

    def compile(self, loss=None, settings=None):
        super().compile(total_loss(self.encoder), settings)