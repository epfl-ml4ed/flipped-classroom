#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import os

import logging
logging.getLogger().setLevel(logging.DEBUG)

class Extractor():

    def __init__(self, name=None):
        self.name = name
        self.features = None
        self.feature_values = None

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

    def exists(self, course, settings):
        filename = settings['timeframe'] + '-' + self.name + '-' + course.course_id.lower().replace('-', '_')
        return os.path.exists(os.path.join(settings['workdir'], filename))

    def save(self, course, settings):
        assert settings['workdir'].endswith('/')

        self.dir = settings['timeframe'] + '-' + self.name + '-' + course.course_id.lower().replace('-', '_')

        if not os.path.exists(os.path.join(settings['workdir'], self.dir)):
            os.makedirs(os.path.join(settings['workdir'], self.dir))

        if hasattr(self, 'feature_values') and self.feature_values is not None:
            # Save the feature values
            np.savez(os.path.join(settings['workdir'], self.dir, 'feature_values.npz'), feature_values=self.feature_values[1])
            # Save the feature labels
            self.feature_values[0].to_csv(os.path.join(settings['workdir'], self.dir, 'feature_labels.csv'), index=False)
            logging.info('saved features {} of shape {} in {}'.format(self.name, self.feature_values[1].shape, self.dir))

        if hasattr(self, 'feature_mapping') and self.feature_mapping is not None:
            # Save the user_index, user_id mappings
            self.feature_mapping.to_csv(os.path.join(settings['workdir'], 'user_id_mapping-' + course.course_id.lower().replace('-', '_') + '.csv'))

        # Save the current settings
        with open(os.path.join(settings['workdir'], self.dir, 'settings.txt'), 'w') as file:
            if hasattr(self, 'features') and self.features is not None:
                file.write(json.dumps({**settings, **{'course_id': course.course_id, 'type': course.type, 'platform': course.platform, 'feature_names': self.get_labels()}}))
            else:
                file.write(json.dumps({**settings, **{'course_id': course.course_id, 'type': course.type, 'platform': course.platform}}))

    def load(self, settings):
        assert settings['feature_set'] is not None
        feature_values = np.load(os.path.join(settings['workdir'], 'feature', settings['feature_set'], 'feature_values.npz'))['feature_values']
        feature_labels = pd.read_csv(os.path.join(settings['workdir'], 'feature', settings['feature_set'], 'feature_labels.csv'))
        self.feature_values = (feature_labels, feature_values)
        with open(os.path.join(settings['workdir'], 'feature', settings['feature_set'], 'settings.txt'), 'rb') as file:
            self.settings = json.load(file)
            if 'feature_list' in settings:
                self.settings['feature_names'] += self.settings['feature_names'][:self.feature_values[1].shape[2]]

    def extract_features(self, data, settings):
        return np.empty()

    def extract_features_bunch(self, course, settings):
        assert 'timeframe' in settings

        if self.exists(course, settings):
            return

        self.save(course, settings)

        filename_logger = os.path.join(settings['workdir'], settings['timeframe'] + '-' + self.name + '-' + course.course_id.lower().replace('-', '_'), 'std.log')
        logging.debug('logs are saved in {}'.format(filename_logger))

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(filename=filename_logger, format='%(asctime)s %(message)s', filemode='w', level=logging.DEBUG)

        data = course.get_clickstream()
        label = course.get_clickstream_grade()
        filter_label_cols = [col for col in label.columns.tolist() if col.startswith('label-')]
        logging.debug('found the column labels {}'.format(filter_label_cols))

        weeks = np.arange(data['week'].max() + 1)
        logging.debug('Found #weeks {}'.format(len(weeks)))
        logging.debug('Detailed weeks:')
        logging.debug(data.groupby(by='week')['date'].max().apply(lambda x: str(x).split(' ')[0]).sort_index())

        users = data['user_id'].unique()

        features_values = []
        features_labels = []
        features_mapping = []

        for u, user_id in tqdm(enumerate(users)):
            user_feature_values = []
            user_feature_labels = label[label['user_id'] == user_id][filter_label_cols].head(1).values.tolist()
            assert len(user_feature_labels) == 1
            for w, week in enumerate(weeks):
                logging.debug('course {} feature_set {} user {} week {}'.format(course.course_id, self.name, user_id, week, settings))
                f = self.extract_features(data[data['user_id'] == user_id], {**settings, **{'week': week, 'course': course}})
                user_feature_values.append(np.array(f))
                logging.debug('Features extracted: {}'.format(np.array(f)))
            features_values.append(np.array(user_feature_values))
            features_labels.append([u] + user_feature_labels[0])
            features_mapping.append(user_id)
        self.feature_values = (pd.DataFrame(features_labels, columns=['user_index'] + filter_label_cols), np.array(features_values))
        self.feature_mapping = pd.DataFrame(features_mapping, columns=['user_id'])
        self.settings = settings
        logging.info('computed features {} of shape {}'.format(self.name, self.feature_values[1].shape))

        self.save(course, settings)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

    def __len__(self):
        assert self.features is not None
        return len(self.features)