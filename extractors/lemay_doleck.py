#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractors.extractor import Extractor
from helpers.feature_extraction import *

import numpy as np

'''
Lemay, D. J., & Doleck, T. (2020). Grade prediction of weekly assignments in MOOCS: mining video-viewing behavior.
Education and Information Technologies, 25(2), 1333-1342.
'''

class LemayDoleck(Extractor):

    def __init__(self, name='base'):
        super().__init__('lemay_doleck')

    def getNbFeatures(self):
        """Returns the number of features"""
        return 10

    def getUserFeatures(self, udata, wid, year):

        features = [
            self.fracSpent(udata),
            self.fracComp(udata),
            self.fracPlayed(udata),
            self.fracPaused(udata),
            self.avgPlayBackRate(udata),
            self.stdPlayBackRate(udata),
            self.totalPause(udata),
            self.totalSeekBackward(udata),
            self.totalSeekBackward(udata),
            self.totalVideoWatcherPerWeek(udata)
        ]

        if len(features) != self.getNbFeatures():
            raise Exception("getNbFeatures is not up-to-date: {len(features)} != {self.getNbFeatures()}")

        return list(np.nan_to_num(features))

    def fracSpent(self, udata):
        """
        @description: The fraction of (real) time the user spent playing the video, relative to its length.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def fracComp(self, udata):
        """
        @description: The percentage of the video that the user played, not counting repeated play position intervals; hence, it must be between 0 and 1.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def fracPlayed(self, udata):
        """
        @description: The amount of the video that the user played, with repetition, relative to its length.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def fracPaused(self, udata):
        """
        @description: The fraction of time the user spent paused on the video, relative to its length.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def avgPlayBackRate(self, udata):
        """
        @description: The time-average of the playback rates selected by the user. The player on Coursera allows rates between 0.75x and 2.0x the default speed.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def stdPlayBackRate(self, udata):
        """
        @description: The standard deviation of the playback rates selected over time.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def totalPause(self, udata):
        """
        @description: The number of times the user paused the video.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return count_actions(udata, 'Video.Pause')

    def totalSeekBackward(self, udata):
        """
        @description: The number of times the user jumped backward in the video.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return count_actions(udata, 'Video.SeekBackward')

    def totalFastForward(self, udata):
        """
        @description: The number of times the user jumped forward in the video.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def totalVideoWatcherPerWeek(self, udata):
        """
        @description: The number of videos viewed per week.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return