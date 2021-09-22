# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from extractor.extractor import Extractor

from extractor.feature.frequency_event import FrequencyEvent
from extractor.feature.fraction_spent import FractionSpent
from extractor.feature.speed_playback import SpeedPlayback
from extractor.feature.count_unique_element import CountUniqueElement

'''
Lemay, D. J., & Doleck, T. (2020). Grade prediction of weekly assignments in MOOCS: mining video-viewing behavior.
Education and Information Technologies, 25(2), 1333-1342.
'''

class LemayDoleck(Extractor):

    def __init__(self, name='base'):
        super().__init__(name)
        self.name = 'lemay_doleck'

    def extract_features(self, data, settings):
        self.features = [FractionSpent(data, {**settings, **{'type': 'Video.Play', 'mode': 'ratio_duration'}}),
                         FractionSpent(data, {**settings, **{'type': 'Video.Pause', 'mode': 'ratio_played'}}),
                         FractionSpent(data, {**settings, **{'type': 'Video.Play', 'mode': 'completed'}}),
                         FractionSpent(data, {**settings, **{'type': 'Video.Play', 'mode': 'spent'}}),
                         FrequencyEvent(data, {**settings, **{'mode': 'total', 'type': 'Video.Pause'}}),
                         FrequencyEvent(data, {**settings, **{'mode': 'total', 'type': 'Video.SeekBackward'}}),
                         FrequencyEvent(data, {**settings, **{'mode': 'total', 'type': 'Video.SeekForward'}}),
                         SpeedPlayback(data, {**settings, **{'ffunc': np.mean}}),
                         SpeedPlayback(data, {**settings, **{'ffunc': np.std}}),
                         CountUniqueElement(data, {**settings, **{'type': 'video'}})]

        features = [f.compute() for f in self.features]
        assert len(features) == self.__len__()
        return features
