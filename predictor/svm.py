#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.svm import SVC, SVR

from predictor.predictor import Predictor

class Svm(Predictor):

    def __init__(self):
        super().__init__('svm')
        self.type = 'sklearn'
        self.depth = 'shallow'

    def build(self, settings):
        assert 'target_type' in settings

        if settings['target_type'] == 'classification':
            self.predictor = SVC(gamma='auto')
        else:
            self.predictor = SVR(gamma='auto')


