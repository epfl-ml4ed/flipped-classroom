#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import json
import os

from extractor.extractor import Extractor

class ExtractorLoader(Extractor):

    def __init__(self):
        super().__init__('extractor_loader')

    def load(self, settings):
        assert 'feature_set' in settings
        if isinstance(settings['feature_set'], list):
            return self.load_ensemble(settings)
        return super().load(settings)

    def load_ensemble(self, settings):
        assert 'feature_set' in settings and isinstance(settings['feature_set'], list)

        self.settings = {}
        feature_labels = None
        feature_values = None
        course_id, course_type, course_platform = None, None, None
        for fs in settings['feature_set']:
            # Merge features
            feature_values_tmp = np.load(os.path.join(settings['workdir'], 'feature', fs, 'feature_values.npz'))['feature_values']
            if feature_values is None:
                feature_values = feature_values_tmp
            else:
                feature_values = np.append(feature_values, feature_values_tmp, axis=2)
            # Check same labels
            feature_labels_tmp = pd.read_csv(os.path.join(settings['workdir'], 'feature', fs, 'feature_labels.csv'))
            assert feature_labels == None or feature_labels.equals(feature_labels_tmp)
            feature_labels = feature_labels_tmp
            # Check same course and other settings
            settings = json.load(open(os.path.join(settings['workdir'], 'feature', settings['feature_set'], 'settings.txt'), 'rb'))
            assert course_id == None or course_id == settings['course_id']
            course_id = settings['course_id']
            assert course_type == None or course_type == settings['type']
            course_type = settings['type']
            assert course_platform == None or course_platform == settings['platform']
            course_platform = settings['platform']

        # Prepare settings and feature values pair as for any other feature set
        self.settings = {'course_id': course_id, 'type': course_type, 'platform': course_platform}
        self.feature_values = (feature_labels, feature_values)

    def save(self, course, settings):
        raise NotImplementedError()

    def extract_features(self, data, settings):
        raise NotImplementedError()

    def extract_features_bunch(self, course, settings):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()