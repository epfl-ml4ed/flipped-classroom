#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractors.extractor import Extractor
import numpy as np

'''
Chen, F., & Cui, Y. (2020). Utilizing Student Time Series Behaviour in Learning Management Systems for Early Prediction of Course Performance.
In Journal of Learning Analytics, 7(2), 1-17.
'''

class ChenCui(Extractor):

    def __init__(self):
        super().__init__('chen_cui')

    def getNbFeatures(self):
        """Returns the number of features"""
        return 11

    def getUserFeatures(self, udata):

        features = [
            self.totalClicks(udata),
            self.numberSessions(udata),
            self.totalTimeAllSessions(udata),
            self.avgSessionTime(udata),
            self.stdSessionTime(udata),
            self.totalClicksWeekdays(udata),
            self.totalClicksWeekends(udata),
            self.ratioClicksWeekdaysWeekends(udata),
            self.totalClicksOnProblems(udata),
            self.totalTimeOnProblems(udata),
            self.stdTimeOnProblems(udata)
        ]

        if len(features) != self.getNbFeatures():
            raise Exception("getNbFeatures is not up-to-date: {len(features)} != {self.getNbFeatures()}")

        return list(np.nan_to_num(features))

    def totalClicks(self, udata):
        """
        @description: The number of total clicks.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def numberSessions(self, udata):
        """
        @description: The number of online sessions
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def totalTimeAllSessions(self, udata):
        """
        @description: The total time for all online sessions.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def avgSessionTime(self, udata):
        """
        @description: The mean of online session time.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def stdSessionTime(self, udata):
        """
        @description: The standard deviation of online session time.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def totalClicksWeekdays(self, udata):
        """
        @description: The number of clicks during weekdays.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def totalClicksWeekends(self, udata):
        """
        @description: The number of clicks during weekends.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def ratioClicksWeekdaysWeekends(self, udata):
        """
        @description: The ratio of weekend to weekday clicks
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def totalClicksOnProblems(self, udata):
        """
        @description: The number of clicks for module “Quiz”.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def totalTimeOnProblems(self, udata):
        """
        @description: the total time on module “Quiz”.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def stdTimeOnProblems(self, udata):
        """
        @description: The standard deviation of time on module “Quiz”.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return