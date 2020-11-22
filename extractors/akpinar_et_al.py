#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractors.extractor import Extractor
import numpy as np

'''
Akpinar, N. J., Ramdas, A., & Acar, U. (2020). Analyzing Student Strategies In Blended Courses Using Clickstream Data.
In Thirteenth International Conference on Educational Data Mining (EDM 2020).
'''

class AkpinarEtAl(Extractor):

    def __init__(self):
        super().__init__('akpinar_et_al')

    def getNbFeatures(self):
        """Returns the number of features"""
        return 12

    def getUserFeatures(self, udata, utypes, ngram=3):

        features = [self.numberSessions(udata), self.totalClicks(udata), self.attendanceVideos(udata), self.attendanceProblems(udata)] + self.countNGrams(udata, utypes, ngram)

        if len(features) != self.getNbFeatures():
            raise Exception("getNbFeatures is not up-to-date: {len(features)} != {self.getNbFeatures()}")

        return list(np.nan_to_num(features))

    def attendanceVideos(self, udata):
        """
        @description: The time spent in watching videos.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def attendanceProblems(self, udata):
        """
        @description: The time spent in playing with problems.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def countNGrams(self, udata, utypes, ngram=3):
        """
        @description: The time spent in playing with problems.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return
