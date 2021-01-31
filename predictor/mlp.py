#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.neural_network import MLPClassifier, MLPRegressor

from predictor.predictor import Predictor

class Mlp(Predictor):

    def __init__(self):
        super().__init__('svm')
        self.type = 'sklearn'
        self.depth = 'shallow'

    def build(self, settings):
        assert 'target_type' in settings

        if settings['target_type'] == 'classification':
            self.predictor = MLPClassifier()
        else:
            self.predictor = MLPRegressor(gamma='auto')


