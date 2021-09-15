#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from extractor.extractor import Extractor

from extractor.feature.total_clicks import TotalClicks
from extractor.feature.text_length import TextLength
from extractor.feature.bool_event import BoolEvent
'''
Test feedback features

Wan et al
- average_length_forum_post
- number_forum_posts
- number_forum_responses

Chen et al
- Number of clicks for module “Forum”

Once tested we can move the features to the respective authors
'''

class Feedback(Extractor):
    def __init__(self, name='base'):
        super().__init__(name)
        self.name = 'feedback'

    def extract_features(self, data, settings):
        self.features = [
        TotalClicks(data, {**settings, **{'type':'Forum'}}),
        TotalClicks(data, {**settings, **{'type':'Forum.Thread.Launch'}}),
        TotalClicks(data, {**settings, **{'type':'Forum.Thread.View'}}),
        TextLength(data, {**settings, **{'type':'Forum.Thread.Launch',
                                         'field':'post_text',
                                          'ffunc': 'avg'}}),
        TextLength(data, {**settings, **{'type':'Forum.Thread.Launch',
                                         'field':'post_text',
                                          'ffunc': 'max'}}),
        BoolEvent(data, {**settings, **{'type':'Forum'}}),
        BoolEvent(data, {**settings, **{'type':'Forum.Thread.Launch'}})
        ]

        features = [f.compute() for f in self.features]
        assert len(features) == self.__len__()
        return features
