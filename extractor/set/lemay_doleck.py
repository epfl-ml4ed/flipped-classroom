#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from extractor.extractor import Extractor

from extractor.feature.frequency_action import FrequencyAction
from extractor.feature.fraction_spent import FractionSpent
from extractor.feature.speed_playback import SpeedPlayback
from extractor.feature.total_video_watched import TotalVideoWatched

'''
Lemay, D. J., & Doleck, T. (2020). Grade prediction of weekly assignments in MOOCS: mining video-viewing behavior.
Education and Information Technologies, 25(2), 1333-1342.
'''

class LemayDoleck(Extractor):

    def __init__(self, name='base'):
        super().__init__(name)
        self.name = 'lemay_doleck'

    def extract_features(self, data, settings):
        self.features = [FrequencyAction(data, {**settings, **{'type':'Video.Play'}}),
                         FrequencyAction(data, {**settings, **{'type':'Video.SeekBackward'}}),
                         FrequencyAction(data, {**settings, **{'type':'Video.SeekForward'}}),
                         FractionSpent(data, settings),
                         FractionSpent(data, {**settings, **{'mode':'Video.Play', 'type': 'perc_time'}}),
                         FractionSpent(data, {**settings, **{'mode':'Video.Play', 'type': 'repeated_perc_time'}}),
                         FractionSpent(data, {**settings, **{'mode':'Video.Pause', 'type': 'repeated_perc_time'}}),
                         SpeedPlayback(data, {**settings, **{'ffunc':np.mean}}),
                         SpeedPlayback(data, {**settings, **{'ffunc':np.std}}),
                         TotalVideoWatched(data, settings)]
        features = [f.compute() for f in self.features]
        assert len(features) == self.__len__()
        return np.nan_to_num(features)
