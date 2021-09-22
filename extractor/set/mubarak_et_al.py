#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from extractor.extractor import Extractor

from extractor.feature.speed_playback import SpeedPlayback
from extractor.feature.fraction_spent import FractionSpent
from extractor.feature.frequency_event import FrequencyEvent

'''
Mubarak, A. A., Cao, H., & Ahmed, S. A. (2020). Predictive learning analytics using deep learning model in MOOCs’ courses videos.
In Education and Information Technologies, 1-22.
'''

class MubarakEtAl(Extractor):

    def __init__(self, name='base'):
        super().__init__(name)
        self.name = 'mubarak_et_al'

    def extract_features(self, data, settings):
        self.features = [FractionSpent(data, {**settings, **{'type': 'Video.Play', 'mode': 'ratio_duration'}}),
                         FractionSpent(data, {**settings, **{'type': 'Video.Pause', 'mode': 'ratio_played'}}),
                         FractionSpent(data, {**settings, **{'type': 'Video.Play', 'mode': 'entirety'}}),
                         FractionSpent(data, {**settings, **{'type': 'Video.Play', 'mode': 'completed'}}),
                         FractionSpent(data, {**settings, **{'type': 'Video.Play', 'mode': 'spent'}}),
                         FractionSpent(data, {**settings, **{'type': 'Video.Seek', 'mode': 'seek_time', 'phase': 'backward'}}),
                         FractionSpent(data, {**settings, **{'type': 'Video.Seek', 'mode': 'seek_time', 'phase': 'forward'}}),
                         FrequencyEvent(data, {**settings, **{'type': 'Video.Play', 'mode': 'relative'}}),
                         FrequencyEvent(data, {**settings, **{'type': 'Video.Pause', 'mode': 'relative'}}),
                         FrequencyEvent(data, {**settings, **{'type': 'Video.Load', 'mode': 'relative'}}),
                         FrequencyEvent(data, {**settings, **{'mode': 'total', 'type': 'Video.SeekBackward'}}),
                         FrequencyEvent(data, {**settings, **{'mode': 'total', 'type': 'Video.SeekForward'}}),
                         SpeedPlayback(data, {**settings, **{'ffunc': np.mean}})]

        features = [f.compute() for f in self.features]
        assert len(features) == self.__len__()
        return features


