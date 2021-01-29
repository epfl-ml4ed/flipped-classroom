#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tqdm import tqdm
import numpy as np
import pickle
import json
import time
import os

class Extractor():

    def __init__(self, name=None):
        self.name = name
        self.features = None
        self.feature_values = None
        self.time = time.strftime('%Y%m%d_%H%M%S')

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
        assert self.feature_values is not None and settings['filepath'].endswith('/')
        if not os.path.exists(os.path.join(settings['filepath'], self.time + '-' + self.name + '-' + course.course_id)):
            os.makedirs(os.path.join(settings['filepath'], self.time + '-' + self.name + '-' + course.course_id))
        with open(os.path.join(settings['filepath'], self.time + '-' + self.name + '-' + course.course_id, 'feature_values.txt'), 'wb') as file:
            pickle.dump(self.feature_values[1], file)
        with open(os.path.join(settings['filepath'], self.time + '-' + self.name + '-' + course.course_id, 'user_ids.txt'), 'wb') as file:
            pickle.dump(self.feature_values[0], file)
        with open(os.path.join(settings['filepath'], self.time + '-' + self.name + '-' + course.course_id, 'settings.txt'), 'w') as file:
            file.write(json.dumps({**settings, **{'course_id': course.course_id, 'type': course.type, 'platform': course.platform}}))

    def load(self, settings):
        assert os.path.join(settings['feature_set']) is not None
        self.feature_values = (pickle.load(open(os.path.join(settings['workdir'], 'feature', settings['feature_set'], 'user_ids.txt'), 'rb')), pickle.load(open(os.path.join(settings['workdir'], 'feature', settings['feature_set'], 'feature_values.txt'), 'rb')))
        self.settings = json.load(open(os.path.join(settings['workdir'], 'feature', settings['feature_set'], 'settings.txt'), 'rb'))

    def extract_features(self, data, settings):
        pass

    def load_features(self, course, settings, timeframe='week'):
        clickstream = course.get_clickstream()
        if timeframe == 'week':
            weeks = np.arange(clickstream['week'].max() - 1)
            users = clickstream['user_id'].unique()
            user_ids = []
            features_values = []
            for u, user_id in tqdm(enumerate(users)):
                user_ids.append(user_id)
                user_feature_values = []
                data = clickstream[clickstream['user_id'] == user_id].sort_values(by='date')
                for w, week in enumerate(weeks):
                    user_feature_values.append(self.extract_features(data, {**settings, **{'timeframe': timeframe, 'week': week, 'course': course}}))
                features_values.append(user_feature_values)
            self.feature_values = (user_ids, np.array(features_values))
            self.settings = settings
            self.save(course, settings)

    def __len__(self):
        assert self.features is not None
        return len(self.features)