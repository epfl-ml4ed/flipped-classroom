#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractors.he_et_al import HeEtAl
import numpy as np

'''
Mbouzao, B., Desmarais, M. C., & Shrier, I. (2020). Early Prediction of Success in MOOC from Video Interaction Features.
In International Conference on Artificial Intelligence in Education (pp. 191-196). Springer, Cham.
'''

class MbouzaoEtAl(HeEtAl):

    def __init__(self, name='base'):
        """
        @description: Returns the identifier associated with this feature set.
        """
        super().__init__('mbouzao_et_al')

    def getNbFeatures(self):
        """
        @description: Returns the number of expected features.
        """
        return 3

    def getUserFeatures(self, udata, wid, year):
        """
        @description: Returns the user features computed from the udata
        """

        features = [
            self.attendanceRate(udata, wid, year),
            self.utilizationRate(udata, wid, year),
            self.watchingIndex(udata, wid, year),
        ]

        if len(features) != self.getNbFeatures():
            raise Exception("getNbFeatures is not up-to-date: {len(features)} != {self.getNbFeatures()}")

        return list(np.nan_to_num(features))

    def watchingIndex(self, udata, wid, year):
        """
        @description: The watch index (WI) is defined as: WI s,c = URs,c × ARs,c
        @requirement:
        """
        return self.utilizationRate(udata, wid, year) * self.attendanceRate(udata, wid, year)