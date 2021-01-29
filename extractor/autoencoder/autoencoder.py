#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import time
import json
import os

class Autoencoder():

    def __init__(self):
        self.name = 'autoencoder'
        self.time = time.strftime('%Y%m%d_%H%M%S')

    def build(self, features=0, obs_per_timestep=0, settings=None):
        self.encoder = None
        self.model = None

    def compile(self, loss=None, settings=None):
        assert self.encoder is not None and self.model is not None
        optimizer = tf.keras.optimizers.Adam(lr=settings['lr'], beta_1=settings['beta_1'], beta_2=settings['beta_2'], epsilon=settings['epsilon'], decay=settings['lr_decay'])
        self.model.compile(loss=loss, optimizer=optimizer)

    def save(self, settings):
        assert self.encoder is not None and self.model is not None and settings['workdir'].endswith('/')
        if not os.path.exists(os.path.join(settings['workdir'], 'feature', self.time + '-' + self.name)):
            os.makedirs(os.path.join(settings['workdir'], 'feature', self.time + '-' + self.name))
        self.encoder.save(os.path.join(settings['workdir'], 'feature', self.time + '-' + self.name, 'encoder.h5'))
        with open(os.path.join(settings['workdir'], 'feature', self.time + '-' + self.name, 'params.txt'), 'w') as file:
            file.write(json.dumps(settings))

    def load(self, settings):
        assert os.path.exists(settings['workdir'])
        self.encoder = tf.keras.models.load_model(os.path.join(settings['workdir'], 'encoder.h5'))

    def train(self, X, settings):
        assert self.model is not None
        self.model.fit(X, X, batch_size=settings['batch'], epochs=settings['epochs'], shuffle=settings['shuffle'], verbose=settings['verbose'])
        self.save(settings)