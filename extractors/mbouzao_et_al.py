#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractors.extractor import Extractor
import numpy as np

'''
Mbouzao, B., Desmarais, M. C., & Shrier, I. (2020). Early Prediction of Success in MOOC from Video Interaction Features.
In International Conference on Artificial Intelligence in Education (pp. 191-196). Springer, Cham.
'''

class MbouzaoEtAl(Extractor):

    def __init__(self):
        super().__init__('mbouzao_et_al')

    def getNbFeatures(self):
        """Returns the number of features"""
        return 3

    def getUserFeatures(self, udata):

        features =  [
            self.attendanceRate(udata),
            self.utilizationRate(udata),
            self.watchingIndex(udata),
        ]

        if len(features) != self.getNbFeatures():
            raise Exception("getNbFeatures is not up-to-date: {len(features)} != {self.getNbFeatures()}")

        return list(np.nan_to_num(features))

    def attendanceRate(self, udata):
        """
        @description: The attendance rate ARs,c of a student s on a given week c since the beginning of the course, is the number of videos that the student played over to the total number of videos up to that period in time of the course schedule.
        @requirement:
        """
        return

    def utilizationRate(self, udata):
        """
        @description: The utilization rate URs,c of a student s on a given week c since the beginning of the course is the proportion of video play time activity of the student over the sum of video lengths for all videos up to week c.
        @requirement:
        """
        return

    def watchingIndex(self, udata):
        """
        @description: The watch index (WI) is defined as: WI s,c = URs,c × ARs,c
        @requirement:
        """
        return