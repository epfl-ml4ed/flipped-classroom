#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import logging

from predictor.layers.attention import Attention
from predictor.loss.supervised_contrastive_loss import SupervisedContrastiveLoss
from predictor.predictor import Predictor

class LstmWithContrastive(Predictor):

    def __init__(self):
        super().__init__('lstm_with_contrastive')
        self.type = 'tf'
        self.depth = 'deep'
        self.hasproba = True

    def build_encoder(self, settings):
        assert 'target_classes' in settings
        # Create encoder
        inp = tf.keras.layers.Input(shape=settings['input_shape'])
        x = tf.keras.layers.LSTM(settings['params_grid']['hidden_units'], return_sequences=True, implementation=1)(inp)
        self.encoder = tf.keras.Model(inputs=[inp], outputs=[x])
        self.encoder.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(lr=settings['params_grid']['learning_rate']), metrics=['accuracy'])

    def build_encoder_with_projection_head(self, settings):
        # Create model
        inputs = tf.keras.layers.Input(shape=self.encoder.input_shape[1:])
        x = Attention()(self.encoder(inputs))
        outputs = tf.keras.layers.Dense(settings['params_grid']['projection_units'], activation='relu')(x)
        self.encoder_with_proj = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Compile model
        self.encoder_with_proj.compile(optimizer=tf.keras.optimizers.Adam(settings['params_grid']['learning_rate']), loss=SupervisedContrastiveLoss(settings['params_grid']['temperature']))

    def build_predictor(self, settings):
        assert 'target_classes' in settings

        # Create model

        inp = tf.keras.Input(shape=self.encoder.input_shape[1:])

        x = Attention()(self.encoder(inp))
        outputs = tf.keras.layers.Dense(settings['params_grid']['projection_units'], activation='relu')(x)
        outputs = tf.keras.layers.GaussianDropout(0.25)(outputs)
        x = tf.keras.layers.Dense(settings['target_classes'], activation='softmax')(outputs)
        model = tf.keras.Model(inputs=[inp], outputs=[x])

        # Compile model
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(lr=settings['params_grid']['learning_rate']), metrics=tf.keras.metrics.SparseCategoricalAccuracy())

        self.predictor = model

    def build(self, settings):
        self.build_encoder(settings)

    def fit(self, X, y, settings):
        logging.info('added the param grid {}'.format(settings['params_grid']))

        X, y = self.prepare_data(X, y, settings)

        # Prepare callback and fit
        cb = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.005, patience=25, verbose=1, mode='min', restore_best_weights=True)
        self.build_encoder_with_projection_head(settings)
        self.encoder_with_proj.fit(X, y, batch_size=settings['params_grid']['batch_size'], epochs=500, shuffle=settings['params_grid']['shuffle'], callbacks=[cb], verbose=settings['params_grid']['verbose'])

        # Prepare callback and fit
        cb = tf.keras.callbacks.EarlyStopping(monitor='sparse_categorical_accuracy', min_delta=0.005, patience=5, verbose=1, mode='max', restore_best_weights=True)
        self.build_predictor(settings)
        self.predictor.fit(X, y, batch_size=settings['params_grid']['batch_size'], epochs=settings['params_grid']['epochs'], shuffle=settings['params_grid']['shuffle'], callbacks=[cb], verbose=settings['params_grid']['verbose'])

    def predict(self, X, settings, proba=False):

        assert self.predictor is not None

        X, _ = self.prepare_data(X, None, settings)

        if proba:
            return self.predictor.predict(X)[:, 1]

        return np.argmax(self.predictor.predict(X), axis=1)