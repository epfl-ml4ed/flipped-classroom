#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.svm import SVC, SVR

from predictor.predictor import Predictor

class Svm(Predictor):

    def __init__(self):
        super().__init__('svm')
        self.type = 'sklearn'
        self.depth = 'shallow'
        self.hasproba = True

    def build(self, settings):
        assert 'target_type' in settings

        if settings['target_type'] == 'classification':
            self.predictor = SVC(random_state=0, probability=True)
        else:
            self.predictor = SVR()

        if 'params_grid' in settings:
            self.add_grid(settings)


