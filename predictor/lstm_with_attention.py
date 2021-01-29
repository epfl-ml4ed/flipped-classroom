#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from sklearn.model_selection import StratifiedKFold
from predictor.attention import Attention
from predictor.predictor import Predictor

class LSTMWithAttention(Predictor):

    def __init__(self):
        super().__init__('lstm_with_attention')

    def build(self, settings):
        inp = tf.keras.layers.Input(shape=settings['input_shape'])
        x = tf.keras.layers.LSTM(settings['hidden_units'], return_sequences=True, implementation=1)(inp)
        x = Attention()(x)
        x = tf.keras.layers.Dense(settings['classes'], activation=settings['activation'])(x)
        self.predictor = tf.keras.Model(inputs=[inp], outputs=[x])

    def compile(self, settings):
        self.predictor.compile(optimizer=tf.keras.optimizers.Adam(settings['lr']), loss=settings['loss'], metrics=settings['metrics'])

    def fit(self, X, y, settings):
        self.predictor.fit(X, y, batch_size=settings['batch'], epochs=settings['epochs'], shuffle=settings['shuffle'], verbose=settings['verbose'])

    def predict(self, X, settings, proba=False):
        assert self.predictor is not None
        if proba:
            return self.predictor.predict(X, batch_size=settings['batch']).flatten()
        return np.round(self.predictor.predict(X, batch_size=settings['batch']).flatten())