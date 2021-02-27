#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os

from routine.detect_best_features import main

logging.getLogger().setLevel(logging.INFO)

if __name__ == "__main__":

    for feature_set in os.listdir('../data/result/edm21/feature'):

        if feature_set.endswith('csv'):
            continue

        main({'task': 'detect_best_features', 'target': 'label-pass-fail', 'target_type': 'classification', 'workdir':'../data/result/edm21/', 'feature_set': feature_set, 'mean_weight': '4.0'})
