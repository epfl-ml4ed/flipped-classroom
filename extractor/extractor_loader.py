#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import logging
import json
import os

from extractor.extractor import Extractor

class ExtractorLoader(Extractor):

    def __init__(self):
        super().__init__('feature-ensemble')

    def load(self, settings):
        assert 'feature_set' in settings or 'feature_list' in settings

        if 'feature_list' in settings:
            self.load_ensemble(settings)
            return

        super().load(settings)

        filepath = os.path.join(settings['workdir'], 'feature', settings['feature_set'], 'feature_selected.txt')

        feature_labels = self.feature_values[0]
        feature_values = self.feature_values[1]

        with open(filepath, 'rb') as file:
            selection = json.load(file)
            if os.path.exists(filepath) and 'selected_features' in settings and settings['selected_features']:
                feature_values = self.feature_values[1][:, :, np.array(selection['support']) > 0]
                logging.info('loaded best-selected features from {} as {}'.format(filepath, np.array(selection['feature_names'])[np.array(selection['support']) > 0]))
            self.feature_names = selection['feature_names']
            logging.info('loaded feature names {}'.format(self.feature_names))

        self.feature_values = (feature_labels, feature_values)

    def load_ensemble(self, settings):
        assert 'feature_list' in settings

        self.settings = {}
        feature_labels = None
        feature_values = None
        course_id, course_type, course_platform, feature_names = None, None, None, []

        for fs in settings['feature_list']:
            logging.info('loading {} feature set'.format(fs))

            # Merge features
            feature_values_tmp = np.load(os.path.join(settings['workdir'], 'feature', fs, 'feature_values.npz'))['feature_values']

            if feature_values is None:
                feature_values = feature_values_tmp
            else:
                feature_values = np.append(feature_values, feature_values_tmp, axis=2)
            logging.info('loaded {} for feature set {}'.format(feature_values.shape, fs))

            # Check same labels
            feature_labels_tmp = pd.read_csv(os.path.join(settings['workdir'], 'feature', fs, 'feature_labels.csv'))
            feature_labels = feature_labels_tmp

            # Check same course and other settings
            feature_settings = json.load(open(os.path.join(settings['workdir'], 'feature', fs, 'settings.txt'), 'rb'))
            course_id = feature_settings['course_id']
            course_type = feature_settings['type']
            course_platform = feature_settings['platform']
            feature_names += [fs.split('-')[1] + '-' + e for e in feature_settings['feature_names'][:feature_values_tmp.shape[2]]]

        # Prepare settings and feature values pair as for any other feature set
        self.settings = {'course_id': course_id, 'type': course_type, 'platform': course_platform, 'feature_names': feature_names}
        self.feature_values = (feature_labels, feature_values)

    def save(self, course, settings, label='ensemble'):
        assert self.feature_values is not None and settings['workdir'].endswith('/')

        filename = settings['timeframe'] + '-' + label + '-' + course

        if not os.path.exists(os.path.join(settings['workdir'], 'feature', filename)):
            os.makedirs(os.path.join(settings['workdir'], 'feature', filename))

        # Save the feature values
        np.savez(os.path.join(settings['workdir'], 'feature', filename, 'feature_values.npz'), feature_values=self.feature_values[1])
        # Save the feature labels
        self.feature_values[0].to_csv(os.path.join(settings['workdir'], 'feature', filename, 'feature_labels.csv'), index=False)
        # Save the current settings
        with open(os.path.join(settings['workdir'], 'feature', filename, 'settings.txt'), 'w') as file:
            file.write(json.dumps(settings))

        logging.info('saved features {} of shape {} in {}'.format(label, self.feature_values[1].shape, filename))

    def extract_features(self, data, settings):
        raise NotImplementedError()

    def extract_features_bunch(self, course, settings):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()