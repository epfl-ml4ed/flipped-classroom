#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractors.extractor import Extractor
import numpy as np

'''
He, H., Zheng, Q., Dong, B., & Yu, H. (2018, July). Measuring Student's Utilization of Video Resources and Its Effect on Academic Performance.
In 2018 IEEE 18th International Conference on Advanced Learning Technologies (ICALT) (pp. 196-198). IEEE.
'''

class HeEtAl(Extractor):

    def __init__(self):
        super().__init__('mbouzao_et_al')

    def getNbFeatures(self):
        """Returns the number of features"""
        return 3

    def getUserFeatures(self, udata):

        features =  [
            self.attendanceRate(udata),
            self.utilizationRate(udata),
            self.watchingRatio(udata),
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

    def watchingRatio(self, udata):
        """
        @description: The student’s overall specialty WR is defined as the ratio between utilization rate and attendance rate. For instance, 1 means that the student s completely watches the video since he/she opened it.
        @requirement:
        """
        return