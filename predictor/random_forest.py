#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from predictor.predictor import Predictor

class RandomForest(Predictor):

    def __init__(self):
        super().__init__('random_forest')

    def build(self, settings):
        assert 'target_type' in settings

        if settings['target_type'] == 'classification':
            self.predictor = RandomForestClassifier()
        else:
            self.predictor = RandomForestRegressor()

