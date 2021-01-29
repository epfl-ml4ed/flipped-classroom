#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestClassifier

from predictor.predictor import Predictor

class RandomForest(Predictor):

    def __init__(self):
        super().__init__('random_forest')

    def build(self, settings):
        self.predictor = RandomForestClassifier()

