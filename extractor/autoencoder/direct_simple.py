#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from extractor.autoencoder.layers import dense_unit
from extractor.autoencoder.autoencoder import Autoencoder

class DirectSimple(Autoencoder):

    def __init__(self):
        super().__init__()
        self.name = 'direct_simple'

    def build(self, features=0, obs_per_timestep=0, settings=None):
        super().build(features, obs_per_timestep, settings)
        shape = [None, features]
    
        # Encoder
        x = tf.keras.layers.Input(shape=(shape[1],))
        r = x
        for _ in range(settings['dense']):
            r = dense_unit(r, settings)
    
        # Normal distribution in latent space
        z = tf.keras.layers.Dense(settings['latent_dim'])(r)
    
        # Decoder
        h = z
        for _ in range(settings['dense']):
            h = dense_unit(h, settings)
    
        # Estimate normal distribution for output features
        x_estimate = tf.keras.layers.Dense(features)(h)
    
        model = tf.keras.models.Model(x, x_estimate)
        model.higgins_beta = tf.keras.layers.Input(tensor=tf.constant(1e-5))
    
        self.encoder = tf.keras.models.Model(inputs=[x], outputs=[z])
        self.latent_statistics = tf.keras.models.Model(inputs=[x], outputs=[z, z])
        self.model = tf.keras.models.Model(inputs=[x], outputs=[x_estimate])

    def compile(self, loss=None, settings=None):
        super().compile('mean_squared_error', settings)