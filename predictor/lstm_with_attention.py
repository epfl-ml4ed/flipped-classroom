#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from predictor.layers.attention import Attention
from predictor.predictor import Predictor

class LstmWithAttention(Predictor):

    def __init__(self):
        super().__init__('lstm_with_attention')
        self.type = 'tf'
        self.depth = 'deep'

    def build(self, settings):
        assert 'classes' in settings

        inp = tf.keras.layers.Input(shape=settings['input_shape'])
        x = tf.keras.layers.Masking(mask_value=-1)(inp)
        x = tf.keras.layers.LSTM(settings['hidden_units'], return_sequences=True, implementation=1)(x)
        x = Attention()(x)

        if settings['classes'] > 1:
            x = tf.keras.layers.Dense(settings['classes'], activation='softmax')(x)
        else:
            x = tf.keras.layers.Dense(settings['classes'], activation='sigmoid')(x)

        self.predictor = tf.keras.Model(inputs=[inp], outputs=[x])

    def compile(self, settings):
        assert 'target_type' in settings

        if settings['target_type'] == 'classification':
            self.predictor.compile(optimizer=tf.keras.optimizers.Adam(settings['lr']), loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            self.predictor.compile(optimizer=tf.keras.optimizers.Adam(settings['lr']), loss='mean_squared_error', metrics=['mse'])

    def fit(self, X, y, settings):
        self.predictor.fit(X, y.astype(np.float), batch_size=settings['batch'], epochs=settings['epochs'], shuffle=settings['shuffle'], verbose=settings['verbose'])

    def predict(self, X, settings, proba=False):
        assert self.predictor is not None
        if proba:
            return self.predictor.predict(X, batch_size=settings['batch']).flatten()
        return np.round(self.predictor.predict(X, batch_size=settings['batch']).flatten())