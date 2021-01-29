#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

class SampleNormal(tf.keras.layers.Layer):

    def __init__(self, epsilon_std=1.0, **kwargs):
        self.epsilon_std = epsilon_std
        super(SampleNormal, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'epsilon_std': self.epsilon_std,
        })
        return config

    def build(self, input_shape):
        super(SampleNormal, self).build(input_shape)

    def call(self, x, mask=None):
        z_mean, z_log_var = x
        epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(z_mean), mean=0., stddev=self.epsilon_std)
        return z_mean + tf.keras.backend.exp(z_log_var / 2) * epsilon

    def get_output_shape_for(self, input_shape):
        return input_shape[0]

def dense_unit(input, config):
    r = tf.keras.layers.Dense(config['inner_dim'], kernel_initializer=config['weight_initialization'])(input)
    r = tf.keras.layers.BatchNormalization()(r)
    r = tf.keras.layers.Activation(config['activation'])(r)
    r = tf.keras.layers.Dropout(config['dropout'])(r)
    return r

def recurrent_unit(input, config):
    r = tf.keras.layers.LSTM(config['inner_dim'], return_sequences=True)(input)
    r = tf.keras.layers.BatchNormalization()(r)
    r = tf.keras.layers.Activation(config['activation'])(r)
    r = tf.keras.layers.Dropout(config['dropout'])(r)
    return r

def convolutional_unit(input, config):
    r = tf.keras.layers.Conv1D(config['conv_filter_n'], 3, padding='same')(input)
    r = tf.keras.layers.BatchNormalization()(r)
    r = tf.keras.layers.Activation(config['activation'])(r)
    r = tf.keras.layers.Dropout(config['dropout'])(r)
    return r

def sampling(args):
    epsilon_std = 1.0
    z_mean, z_log_var = args
    epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(z_mean), mean=0., std=epsilon_std)
    return z_mean + tf.keras.backend.exp(z_log_var / 2) * epsilon