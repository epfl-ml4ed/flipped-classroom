#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from extractor.extractor import Extractor

from extractor.feature.total_clicks import TotalClicks
from extractor.feature.seek_length import SeekLength
from extractor.feature.frequency_event import FrequencyEvent
from extractor.feature.weekly_prop import WeeklyProp
from extractor.feature.pause_duration import PauseDuration
from extractor.feature.time_speeding_up import TimeSpeedingUp

'''
Lallé, S., & Conati, C. (2020). A Data-Driven Student Model to Provide Adaptive Support During Video Watching Across MOOCs.
In International Conference on Artificial Intelligence in Education (pp. 282-295). Springer, Cham.
'''

class LalleConati(Extractor):

    def __init__(self, name='base'):
        super().__init__(name)
        self.name = 'lalle_conati'

    def extract_features(self, data, settings):
        self.features = [TotalClicks(data, {**settings, **{'type': 'Video.Load'}}),
                         WeeklyProp(data, {**settings, **{'type':'watched', 'ffunc': np.mean}}),
                         WeeklyProp(data, {**settings, **{'type':'watched', 'ffunc': np.std}}),
                         WeeklyProp(data, {**settings, **{'type':'replayed', 'ffunc': np.mean}}),
                         WeeklyProp(data, {**settings, **{'type':'replayed', 'ffunc': np.std}}),
                         WeeklyProp(data, {**settings, **{'type':'interrupted', 'ffunc': np.mean}}),
                         WeeklyProp(data, {**settings, **{'type':'interrupted', 'ffunc': np.std}}),
                         TotalClicks(data, {**settings, **{'type': 'Video'}}),
                         FrequencyEvent(data, {**settings, **{'type':'Video'}}),
                         FrequencyEvent(data, {**settings, **{'type':'Video.Load'}}),
                         FrequencyEvent(data, {**settings, **{'type':'Video.Play'}}),
                         FrequencyEvent(data, {**settings, **{'type':'Video.Pause'}}),
                         FrequencyEvent(data, {**settings, **{'type':'Video.Stop'}}),
                         FrequencyEvent(data, {**settings, **{'type':'Video.SeekBackward'}}),
                         FrequencyEvent(data, {**settings, **{'type':'Video.SeekForward'}}),
                         FrequencyEvent(data, {**settings, **{'type':'Video.SpeedChange'}}),
                         SeekLength(data, {**settings, **{'ffunc': np.mean}}),
                         SeekLength(data, {**settings, **{'ffunc': np.std}}),
                         PauseDuration(data, {**settings, **{'ffunc': np.mean}}),
                         PauseDuration(data, {**settings, **{'ffunc': np.std}}),
                         TimeSpeedingUp(data, {**settings, **{'ffunc': np.mean}}),
                         TimeSpeedingUp(data, {**settings, **{'ffunc': np.std}})]
        features = [f.compute() for f in self.features]
        assert len(features) == self.__len__()
        return features