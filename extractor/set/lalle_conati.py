#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from extractor.extractor import Extractor

from extractor.feature.total_clicks import TotalClicks
from extractor.feature.total_views import TotalViews
from extractor.feature.time_speeding_up import TimeSpeedingUp
from extractor.feature.seek_length import SeekLength
from extractor.feature.frequency_action import FrequencyAction
from extractor.feature.weekly_prop import WeeklyProp

'''
Lallé, S., & Conati, C. (2020, July). A Data-Driven Student Model to Provide Adaptive Support During Video Watching Across MOOCs.
In International Conference on Artificial Intelligence in Education (pp. 282-295). Springer, Cham.
'''

class LalleConati(Extractor):

    def __init__(self, name='base'):
        super().__init__(name)
        self.name = 'lalle_conati'

    def extract_features(self, data, settings):
        self.features = [TotalClicks(data, settings),
                         TotalViews(data, {**settings, **{'type': 'video'}}),
                         SeekLength(data, {**settings, **{'ffunc':np.mean}}),
                         SeekLength(data, {**settings, **{'ffunc':np.std}}),
                         FrequencyAction(data, {**settings, **{'type':'Video.Play'}}),
                         FrequencyAction(data, {**settings, **{'type':'Video.Pause'}}),
                         FrequencyAction(data, {**settings, **{'type':'Video.Stop'}}),
                         FrequencyAction(data, {**settings, **{'type':'Video.SeekBackward'}}),
                         FrequencyAction(data, {**settings, **{'type':'Video.SeekForward'}}),
                         FrequencyAction(data, {**settings, **{'type':'Video.SpeedChange'}}),
                         WeeklyProp(data, {**settings, **{'type':'watched', 'ffunc': np.mean}}),
                         WeeklyProp(data, {**settings, **{'type':'watched', 'ffunc': np.std}}),
                         WeeklyProp(data, {**settings, **{'type':'replayed', 'ffunc': np.mean}}),
                         WeeklyProp(data, {**settings, **{'type':'replayed', 'ffunc': np.std}}),
                         WeeklyProp(data, {**settings, **{'type':'interrupted', 'ffunc': np.mean}}),
                         WeeklyProp(data, {**settings, **{'type':'interrupted', 'ffunc': np.std}}),
                         FrequencyAction(data, settings)]
        features = [f.compute() for f in self.features]
        assert len(features) == self.__len__()
        return np.nan_to_num(features)