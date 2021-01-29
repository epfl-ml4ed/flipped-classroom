#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from extractor.autoencoder.autoencoder import Autoencoder
from extractor.autoencoder.losses import vae_loss

class L1Norm(Autoencoder):

    def __init__(self):
        super().__init__()
        self.name = 'l1_norm'

    def build(self, features=0, obs_per_timestep=0, settings=None):
        super().build(features, obs_per_timestep, settings)
        n_timesteps = int(features / obs_per_timestep)
        shape = [None, n_timesteps, obs_per_timestep]
        rnn = tf.keras.layers.LSTM

        # Encoder
        x = tf.keras.layers.Input(shape=(shape[1], shape[2]))
        h = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(settings['inner_dim']), input_shape=(shape[1], shape[2]))(x)
        r = rnn(settings['inner_dim'], input_shape=(None, settings['inner_dim']))(h)
        z = tf.keras.layers.Dense(settings['latent_dim'])(r)

        # Decoder
        z_repeated = tf.keras.layers.RepeatVector(shape[1])(z)
        d = rnn(settings['inner_dim'], return_sequences=True)(z_repeated)
        x_estimate = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(settings['output_dim']))(d)

        self.encoder = tf.keras.models.Model(inputs=[x], outputs=[z])
        self.model = tf.keras.models.Model(inputs=[x], outputs=[x_estimate])

    def compile(self, loss=None, settings=None):
        super().compile(vae_loss(self.encoder), settings)