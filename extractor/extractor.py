#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tqdm import tqdm
import pandas as pd
import numpy as np
import logging
import json
import time
import os

class Extractor():

    def __init__(self, name=None):
        self.name = name
        self.features = None
        self.feature_values = None
        self.time = time.strftime('%Y%m%d-%H%M%S')

    def get_name(self):
        assert self.name is not None
        return self.name

    def get_labels(self):
        assert self.features is not None
        return [f.get_name() for f in self.features]

    def get_features_objects(self):
        assert self.features is not None
        return self.features

    def get_settings(self):
        assert self.settings is not None
        return self.settings

    def get_features_values(self):
        assert self.feature_values is not None
        return self.feature_values

    def save(self, course, settings):
        assert self.feature_values is not None and settings['workdir'].endswith('/')

        course_id_pseudo = course.course_id.lower().replace('-', '_')

        if not os.path.exists(os.path.join(settings['workdir'], self.settings['timeframe'] + '-' + self.name + '-' + course_id_pseudo + '-' + self.time)):
            os.makedirs(os.path.join(settings['workdir'], self.settings['timeframe'] + '-' + self.name + '-' + course_id_pseudo + '-' + self.time))

        # Save the feature values
        np.savez(os.path.join(settings['workdir'], self.settings['timeframe'] + '-' + self.name + '-' + course_id_pseudo + '-' + self.time, 'feature_values.npz'), feature_values=self.feature_values[1])
        # Save the feature labels
        self.feature_values[0].to_csv(os.path.join(settings['workdir'], self.settings['timeframe'] + '-' + self.name + '-' + course_id_pseudo + '-' + self.time, 'feature_labels.csv'), index=False)
        # Save the current settings
        with open(os.path.join(settings['workdir'], self.settings['timeframe'] + '-' + self.name + '-' + course_id_pseudo + '-' + self.time, 'settings.txt'), 'w') as file:
            file.write(json.dumps({**settings, **{'course_id': course.course_id, 'type': course.type, 'platform': course.platform}}))

        logging.info('saved features {} of shape {}'.format(self.name, self.feature_values[1].shape))

    def load(self, settings):
        assert settings['feature_set'] is not None
        feature_values = np.load(os.path.join(settings['workdir'], 'feature', settings['feature_set'], 'feature_values.npz'))['feature_values']
        feature_labels = pd.read_csv(os.path.join(settings['workdir'], 'feature', settings['feature_set'], 'feature_labels.csv'))
        self.feature_values = (feature_labels, feature_values)
        self.settings = json.load(open(os.path.join(settings['workdir'], 'feature', settings['feature_set'], 'settings.txt'), 'rb'))

    def extract_features(self, data, settings):
        return np.empty()

    def extract_features_bunch(self, course, settings):
        assert 'timeframe' in settings

        data = course.get_clickstream()
        label = course.get_clickstream_grade()
        filter_label_cols = [col for col in label.columns.tolist() if col.startswith('label-')]
        logging.debug('found the column labels {}'.format(filter_label_cols))

        weeks = np.arange(data['week'].max() - 1)
        users = data['user_id'].unique()

        features_values = []
        features_labels = []

        for u, user_id in tqdm(enumerate(users)):
            user_feature_values = []
            user_feature_labels = label[label['user_id'] == user_id][filter_label_cols].head(1).values.tolist()
            assert len(user_feature_labels) == 1
            for w, week in enumerate(weeks):
                f = self.extract_features(data[data['user_id'] == user_id], {**settings, **{'week': week, 'course': course}})
                user_feature_values.append(np.array(f))
            features_values.append(np.array(user_feature_values))
            features_labels.append([u] + user_feature_labels[0])
        self.feature_values = (pd.DataFrame(features_labels, columns=['user_index'] + filter_label_cols), np.array(features_values))
        self.settings = settings
        logging.info('computed features {} of shape {}'.format(self.name, self.feature_values[1].shape))

        self.save(course, settings)

    def __len__(self):
        assert self.features is not None
        return len(self.features)