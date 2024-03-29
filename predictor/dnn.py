#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import logging

from predictor.predictor import Predictor

class Dnn(Predictor):

    def __init__(self):
        super().__init__('dnn')
        self.type = 'tf'
        self.depth = 'deep'
        self.hasproba = True

    def build(self, settings):
        assert 'target_classes' in settings

        # Create layers
        input = tf.keras.layers.Input(shape=settings['input_shape'])
        x = tf.keras.layers.Dense(settings['params_grid']['hidden_units'], activation='relu')(input)
        x = tf.keras.layers.Dropout(settings['params_grid']['dropout_rate'])(x)
        x = tf.keras.layers.Dense(settings['target_classes'], activation='softmax')(x)
        model = tf.keras.models.Model(inputs=[input], outputs=[x])

        # Compile model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=settings['params_grid']['learning_rate']), metrics=['accuracy'])

        self.predictor = model

    def fit(self, X, y, settings):
        logging.info('added the param grid {}'.format(settings['params_grid']))

        X, y = self.prepare_data(X, y, settings)

        # Prepare callback and fit
        self.predictor.fit(X, y, batch_size=settings['params_grid']['batch_size'], epochs=settings['params_grid']['epochs'], shuffle=settings['params_grid']['shuffle'], verbose=settings['params_grid']['verbose'])

    def predict(self, X, settings, proba=False):

        assert self.predictor is not None

        X, _ = self.prepare_data(X, None, settings)

        if proba:
            return self.predictor.predict(X)[:, 1]

        return np.argmax(self.predictor.predict(X), axis=1)