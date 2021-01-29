#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from extractor.autoencoder.layers import dense_unit, SampleNormal
from extractor.autoencoder.autoencoder import Autoencoder
from extractor.autoencoder.losses import vae_loss_basic

class VariationalSimple(Autoencoder):

    def __init__(self):
        super().__init__()
        self.name = 'vae_d2_lstam'

    def build(self, features=0, obs_per_timestep=0, settings=None):
        super().build(features, obs_per_timestep, settings)
        shape = [None, features]

        # Encoder
        x = tf.keras.layers.Input(shape=(shape[1],))
        r = x
        for _ in range(settings['dense']):
            r = dense_unit(r, settings)

        # Normal distribution in latent space
        z_mean = tf.keras.layers.Dense(settings['latent_dim'])(r)
        z_log_var = tf.keras.layers.Dense(settings['latent_dim'], kernel_initializer='zero')(r)
        z = SampleNormal()([z_mean, z_log_var])

        # Decoder
        h = z
        for _ in range(settings['dense']):
            h = dense_unit(h, settings)

        # Estimate normal distribution for output features
        x_estimate = tf.keras.layers.Dense(features)(h)
        x_log_var = tf.keras.layers.Dense(features)(h)

        self.z_mean_m = tf.keras.models.Model(inputs=[x], outputs=[z_mean])
        self.z_log_var_m = tf.keras.models.Model(inputs=[x], outputs=[z_log_var])
        self.x_log_var_m = tf.keras.models.Model(inputs=[x], outputs=[x_log_var])

        self.encoder = tf.keras.models.Model(inputs=[x], outputs=[z_mean])
        self.latent_statistics = tf.keras.models.Model(inputs=[x], outputs=[z_mean, z_log_var])
        self.model = tf.keras.models.Model(inputs=[x], outputs=[x_estimate])

    def compile(self, loss=None, settings=None):
        super().compile(vae_loss_basic(self.z_mean_m, self.z_log_var_m, self.x_log_var_m, settings['beta']), settings)