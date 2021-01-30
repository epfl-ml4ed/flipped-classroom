#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from extractor.extractor import Extractor

from extractor.feature.attendance_rate import AttendanceRate
from extractor.feature.utilization_rate import UtilizationRate
from extractor.feature.watching_index import WatchingIndex

'''
Mbouzao, B., Desmarais, M. C., & Shrier, I. (2020). Early Prediction of Success in MOOC from Video Interaction Features.
In International Conference on Artificial Intelligence in Education (pp. 191-196). Springer, Cham.
'''

class MbouzaoEtAl(Extractor):

    def __init__(self, name='base'):
        super().__init__(name)
        self.name = 'mbouzao_et_al'

    def extract_features(self, data, settings):
        self.features = [AttendanceRate(data, settings),
                         UtilizationRate(data, settings),
                         WatchingIndex(data, settings)]

        features = [f.compute() for f in self.features]
        assert len(features) == self.__len__()
        return features
