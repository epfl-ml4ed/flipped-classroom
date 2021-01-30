#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

from extractor.extractor import Extractor

from extractor.feature.reg_peak_time import RegPeakTime
from extractor.feature.reg_weekly_sim import RegWeeklySim
from extractor.feature.reg_periodicity import RegPeriodicity
from extractor.feature.delay_lecture import DelayLecture

'''
Boroujeni, M. S., Sharma, K., Kidziński, Ł., Lucignano, L., & Dillenbourg, P. (2016). How to quantify student’s regularity?
In European Conference on Technology Enhanced Learning (pp. 277-291). Springer, Cham.
'''

class BoroujeniEtAl(Extractor):
    def __init__(self, name='base'):
        super().__init__(name)
        self.name = 'boroujeni_et_al'

    def extract_features(self, data, settings):
        self.features = [RegPeakTime(data, {**settings, **{'mode': 'dayhour'}}),
                         RegPeriodicity(data, {**settings, **{'mode': 'm1'}}),
                         DelayLecture(data, settings)]

        if self.settings['timeframe'] is not 'eq-week' and self.settings['week'] > 0:
            logging.info('computing also inter-week features for {}'.format(self.name))
            self.features += [RegPeakTime(data, {**settings, **{'mode': 'weekday'}}),
                              RegWeeklySim(data, {**settings, **{'mode': 'm1'}}),
                              RegWeeklySim(data, {**settings, **{'mode': 'm2'}}),
                              RegWeeklySim(data, {**settings, **{'mode': 'm3'}}),
                              RegPeriodicity(data, {**settings, **{'mode': 'm2'}}),
                              RegPeriodicity(data, {**settings, **{'mode': 'm3'}})]

        features = [f.compute() for f in self.features]
        assert len(features) == self.__len__()
        return features