#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from extractor.autoencoder.layers import SampleNormal
from extractor.autoencoder.autoencoder import Autoencoder
from extractor.autoencoder.losses import vae_loss_basic

class Variational(Autoencoder):

    def __init__(self):
        super().__init__()
        self.name = 'variational'

    def build(self, features=0, obs_per_timestep=0, settings=None):
        super().build(features, obs_per_timestep, settings)
        n_timesteps = int(features / obs_per_timestep)
        shape = [None, n_timesteps, obs_per_timestep]
        rnn = tf.keras.layers.LSTM

        # Encoder
        x = tf.keras.layers.Input(shape=(shape[1], shape[2]))
        h = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(settings['inner_dim']), input_shape=(shape[1], shape[2]))(x)
        h = tf.keras.layers.BatchNormalization()(h)
        for _ in range(settings['layers']):
            h = rnn(settings['inner_dim'], return_sequences=True)(h)
            h = tf.keras.layers.BatchNormalization()(h)
        r = rnn(settings['inner_dim'], input_shape=(None, settings['inner_dim']))(h)

        r = tf.keras.layers.BatchNormalization()(r)
        for _ in range(settings['dense']):
            r = tf.keras.layers.Dense(settings['inner_dim'], activation='relu')(r)
            r = tf.keras.layers.BatchNormalization()(r)

        z_mean = tf.keras.layers.Dense(settings['latent_dim'])(r)
        z_log_var = tf.keras.layers.Dense(settings['latent_dim'])(r)
        z = SampleNormal()([z_mean, z_log_var])

        hd = z
        for _ in range(settings['n_dense']):
            hd = tf.keras.layers.Dense(settings['inner_dim'], activation='relu')(hd)
            hd = tf.keras.layers.BatchNormalization()(hd)

        # Decoder
        z_repeated = tf.keras.layers.RepeatVector(shape[1])(hd)
        for i in range(settings['n_layers']):
            z_repeated = rnn(settings['inner_dim'], return_sequences=True)(z_repeated)
            z_repeated = tf.keras.layers.BatchNormalization()(z_repeated)
        d = rnn(settings['inner_dim'], return_sequences=True)(z_repeated)
        d = tf.keras.layers.BatchNormalization()(d)
        x_estimate = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(settings['output_dim']))(d)
        x_log_var = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(settings['output_dim']))(d)

        self.z_mean_m = tf.keras.models.Model(inputs=[x], outputs=[z_mean])
        self.z_log_var_m = tf.keras.models.Model(inputs=[x], outputs=[z_log_var])
        self.x_log_var_m = tf.keras.models.Model(inputs=[x], outputs=[x_log_var])

        self.encoder = tf.keras.models.Model(inputs=[x], outputs=[z_mean])
        self.latent_statistics = tf.keras.models.Model(inputs=[x], outputs=[z_mean, z_log_var])
        self.model = tf.keras.models.Model(inputs=[x], outputs=[x_estimate])

    def compile(self, loss=None, settings=None):
        super().compile(vae_loss_basic(self.z_mean_m, self.z_log_var_m, self.x_log_var_m, settings['beta']), settings)