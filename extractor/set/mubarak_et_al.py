#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from extractor.extractor import Extractor

from extractor.feature.frequency_action import FrequencyAction
from extractor.feature.fraction_spent import FractionSpent
from extractor.feature.time_playback import TimePlayback
from extractor.feature.time_playback_change import TimePlaybackChange

'''
Mubarak, A. A., Cao, H., & Ahmed, S. A. (2020). Predictive learning analytics using deep learning model in MOOCs’ courses videos.
In Education and Information Technologies, 1-22.
'''

class MubarakEtAl(Extractor):

    def __init__(self, name='base'):
        super().__init__(name)
        self.name = 'mubarak_et_al'

    def extract_features(self, data, settings):
        self.features = [FrequencyAction(data, {**settings, **{'type':'Video.Play'}}),
                         FrequencyAction(data, {**settings, **{'type':'Video.Pause'}}),
                         FrequencyAction(data, {**settings, **{'type':'Video.SeekBackward'}}),
                         FrequencyAction(data, {**settings, **{'type':'Video.SeekForward'}}),
                         FrequencyAction(data, {**settings, **{'type':'Video.Load'}}),
                         FractionSpent(data, settings),
                         FractionSpent(data, {**settings, **{'mode':'Video.Play', 'type': 'perc_time'}}),
                         FractionSpent(data, {**settings, **{'mode':'Video.Pause', 'type': 'repeated_perc_time'}}),
                         FractionSpent(data, {**settings, **{'mode':'Video.Play', 'type': 'repeated_perc_time'}}),
                         FractionSpent(data, {**settings, **{'mode':'Video.Play', 'type': 'perc_time_entire_video'}}),
                         TimePlayback(data, {**settings, **{'mode':'Video.Forward', 'ffunc': np.mean}}),
                         TimePlayback(data, {**settings, **{'mode':'Video.Backward', 'ffunc': np.mean}}),
                         TimePlaybackChange(data, {**settings, **{'ffunc':np.mean}})]
        features = [f.compute() for f in self.features]
        assert len(features) == self.__len__()
        return np.nan_to_num(features)


