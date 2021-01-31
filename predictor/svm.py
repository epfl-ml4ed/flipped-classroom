#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.svm import SVC, SVR

from predictor.predictor import Predictor

class SVM(Predictor):

    def __init__(self):
        super().__init__('svm')

    def build(self, settings):
        assert 'target_type' in settings

        if settings['target_type'] == 'classification':
            self.predictor = SVC(gamma='auto')
        else:
            self.predictor = SVR(gamma='auto')


