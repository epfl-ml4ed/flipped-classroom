#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from predictor.loss.supervised_contrastive_loss import SupervisedContrastiveLoss
from predictor.attention import Attention
from predictor.predictor import Predictor

class LSTMWithContrastive(Predictor):

    def __init__(self):
        super().__init__('lstm_with_contrastive')

    def build_encoder(self, settings):
        inp = tf.keras.layers.Input(shape=settings['input_shape'])
        x = tf.keras.layers.LSTM(settings['hidden_units'], return_sequences=True, implementation=1)(inp)
        x = Attention()(x)
        x = tf.keras.layers.Dense(settings['classes'], activation=settings['classes_activation'])(x)
        self.encoder = tf.keras.Model(inputs=[inp], outputs=[x])

    def build_encoder_with_projection_head(self, settings):
        inputs = tf.keras.layers.Input(shape=self.encoder.input_shape[1:])
        features = self.encoder(inputs)
        outputs = tf.keras.layers.Dense(settings['projection_units'], activation='relu')(features)
        outputs = tf.keras.layers.Dense(settings['projection_units'] // 2, activation='relu')(outputs)
        self.encoder_with_proj = tf.keras.Model(inputs=inputs, outputs=outputs)

    def build_predictor(self, settings):
        for layer in self.encoder.layers:
            layer.trainable = settings['trainable']

        inp = tf.keras.Input(shape=self.encoder.input_shape[1:])
        x = tf.keras.layers.Dense(settings['classes'], activation=settings['activation'])(self.encoder(inp))
        self.model = tf.keras.Model(inputs=[inp], outputs=[x])

    def build(self, settings):
        self.build_encoder(settings)
        self.build_encoder_with_projection_head(settings)
        self.build_predictor({'trainable': False})

    def compile(self, settings):
        self.encoder_with_proj.compile(optimizer=tf.keras.optimizers.Adam(settings['lr']), loss=SupervisedContrastiveLoss(settings['temperature']))
        self.predictor.compile(optimizer=tf.keras.optimizers.Adam(settings['lr']), loss=settings['loss'], metrics=settings['metrics'])

    def train(self, X, y, settings):
        self.encoder_with_proj.fit(x=X, y=y, batch_size=settings['batch'], epochs=settings['epochs'], shuffle=settings['shuffle'], verbose=settings['verbose'])
        es_predictor = tf.keras.callbacks.EarlyStopping(monitor='binary_accuracy', verbose=0, patience=settings['patience'], min_delta=settings['min_delta'], mode=settings['mode'], restore_best_weights=True)
        self.predictor.fit(x=X, y=y, batch_size=settings['batch'], epochs=settings['epochs'], shuffle=settings['shuffle'], callbacks=[es_predictor], verbose=settings['verbose'])

    def predict(self, X, settings):
        assert self.predictor is not None
        return self.predictor.predict(X, batch_size=settings['batch']).flatten()