#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from extractor.autoencoder.layers import recurrent_unit, convolutional_unit, SampleNormal
from extractor.autoencoder.autoencoder import Autoencoder
from extractor.autoencoder.losses import vae_loss_basic

class Conv2LSTM1D(Autoencoder):

    def __init__(self):
        super().__init__()
        self.name = 'vae_conv2_lstm_1d'

    def build(self, features=0, obs_per_timestep=0, settings=None):
        super().build(features, obs_per_timestep, settings)
        n_timesteps = int(features / obs_per_timestep)
        shape = [None, features]

        # Encoder
        x = tf.keras.layers.Input(shape=(shape[1],))
        r = tf.keras.layers.Reshape((n_timesteps, obs_per_timestep))(x)
        for _ in range(settings['dense']):
            r = convolutional_unit(r, settings)
        r = tf.keras.layers.Flatten()(r)

        z_mean = tf.keras.layers.Dense(settings['latent_dim'])(r)
        z_log_var = tf.keras.layers.Dense(settings['latent_dim'], kernel_initializer='zero')(r)
        z = SampleNormal()([z_mean, z_log_var])

        # Decoder
        z_repeated = tf.keras.layers.RepeatVector(n_timesteps)(z)
        hd = z_repeated
        for _ in range(settings['layers']):
            hd = recurrent_unit(z_repeated, settings)

        # Estimate normal distribution for output features
        x_estimate_t = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(obs_per_timestep))(hd)
        x_log_var_t = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(obs_per_timestep))(hd)

        # Flatten the output
        x_estimate = tf.keras.layers.Flatten()(x_estimate_t)
        x_log_var = tf.keras.layers.Flatten()(x_log_var_t)

        self.z_mean_m = tf.keras.models.Model(inputs=[x], outputs=[z_mean])
        self.z_log_var_m = tf.keras.models.Model(inputs=[x], outputs=[z_log_var])
        self.x_log_var_m = tf.keras.models.Model(inputs=[x], outputs=[x_log_var])

        self.encoder = tf.keras.models.Model(inputs=[x], outputs=[z_mean])
        self.latent_statistics = tf.keras.models.Model(inputs=[x], outputs=[z_mean, z_log_var])
        self.model = tf.keras.models.Model(inputs=[x], outputs=[x_estimate])

    def compile(self, loss=None, settings=None):
        super().compile(vae_loss_basic(self.z_mean_m, self.z_log_var_m, self.x_log_var_m, settings['beta']), settings)