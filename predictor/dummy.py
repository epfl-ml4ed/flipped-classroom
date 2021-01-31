#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.dummy import DummyClassifier, DummyRegressor

from predictor.predictor import Predictor

class Dummy(Predictor):

    def __init__(self):
        super().__init__('dummy')
        self.type = 'sklearn'
        self.depth = 'shallow'
        self.hasproba = False

    def build(self, settings):
        assert 'target_type' in settings

        if settings['target_type'] == 'classification':
            self.predictor = DummyClassifier(strategy=settings['strategy'] if 'strategy' in settings else 'uniform')
        else:
            self.predictor = DummyRegressor(strategy=settings['strategy'] if 'strategy' in settings else 'mean')

