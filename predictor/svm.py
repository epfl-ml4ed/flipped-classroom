#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.svm import SVC

from predictor.predictor import Predictor

class SVM(Predictor):

    def __init__(self):
        super().__init__('svm')

    def build(self, settings):
        self.predictor = SVC(gamma='auto')

