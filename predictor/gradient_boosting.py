#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from predictor.predictor import Predictor

class GradientBoosting(Predictor):

    def __init__(self):
        super().__init__('gradient_boosting')
        self.type = 'sklearn'
        self.depth = 'shallow'
        self.hasproba = True

    def build(self, settings):
        assert 'target_type' in settings

        if settings['target_type'] == 'classification':
            logging.info('built {} classifier'.format(self.name))
            self.predictor = GradientBoostingClassifier(random_state=0)
        else:
            logging.info('built {} regressor'.format(self.name))
            self.predictor = GradientBoostingRegressor(random_state=0)

        if 'params_grid' in settings:
            self.add_grid(settings)

