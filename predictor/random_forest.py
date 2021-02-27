#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from predictor.predictor import Predictor

class RandomForest(Predictor):

    def __init__(self):
        super().__init__('random_forest')
        self.type = 'sklearn'
        self.depth = 'shallow'
        self.hasproba = True

    def build(self, settings):
        super().build(settings)

        if settings['target_type'] == 'classification':
            logging.info('built {} classifier'.format(self.name))
            self.predictor = RandomForestClassifier(random_state=0)
        else:
            logging.info('built {} regressor'.format(self.name))
            self.predictor = RandomForestRegressor(random_state=0)

        if 'params_grid' in settings:
            self.add_grid(settings)

