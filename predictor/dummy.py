#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

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
            logging.info('built {} classifier'.format(self.name))
            self.predictor = DummyClassifier(random_state=0)
        else:
            logging.info('built {} regressor'.format(self.name))
            self.predictor = DummyRegressor()

        if 'params_grid' in settings:
            self.add_grid(settings)
