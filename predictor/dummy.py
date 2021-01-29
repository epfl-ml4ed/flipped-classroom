#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.dummy import DummyClassifier

from predictor.predictor import Predictor

class Dummy(Predictor):

    def __init__(self):
        super().__init__('dummy')

    def build(self, settings):
        self.predictor = DummyClassifier(strategy=settings['strategy'] if 'strategy' in settings else 'uniform')

