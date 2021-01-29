#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from extractor.extractor import Extractor

from extractor.feature.attendance_rate import AttendanceRate
from extractor.feature.utilization_rate import UtilizationRate
from extractor.feature.watching_ratio import WatchingRatio

'''
He, H., Zheng, Q., Dong, B., & Yu, H. (2018). Measuring Student's Utilization of Video Resources and Its Effect on Academic Performance.
In 2018 IEEE 18th International Conference on Advanced Learning Technologies (ICALT) (pp. 196-198). IEEE.
'''

class HeEtAl(Extractor):

    def __init__(self, name='base'):
        super().__init__(name)
        self.name = 'he_et_al'

    def extract_features(self, data,settings):
        self.features = [AttendanceRate(data, settings),
                         UtilizationRate(data, settings),
                         WatchingRatio(data, settings)]
        features = [f.compute() for f in self.features]
        assert len(features) == self.__len__()
        return np.nan_to_num(features)