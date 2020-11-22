#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractors.extractor import Extractor
from helpers.time import string2Datetime
from helpers.db_query import *

import numpy as np
import sys

'''
Chen, F., & Cui, Y. (2020). Utilizing Student Time Series Behaviour in Learning Management Systems for Early Prediction of Course Performance.
In Journal of Learning Analytics, 7(2), 1-17.
'''

class ChenCui(Extractor):

    def __init__(self, name='base'):
        """
        @description: Returns the identifier associated with this feature set.
        """
        super().__init__('chen_cui')

    def getNbFeatures(self):
        """
        @description: Returns the number of expected features.
        """
        return 11

    def getUserFeatures(self, udata, wid, year):
        """
        @description: Returns the user features computed from the udata
        """

        udata = udata.copy()
        udata['TimeStamp'] = udata['Date']
        udata = udata.sort_values(by='TimeStamp')
        udata['Weekday'] = udata['Date'].apply(lambda x: 1 if x.weekday() < 5 else 0)

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
        return len(udata.index)

    def numberSessions(self, udata):
        """
        @description: The number of online sessions
        @requirement: VideoID, Date (datetime object), EventType
        """
        return len(getSessions(udata, maxSessionLength=120, minNoActions=3).index)

    def totalTimeAllSessions(self, udata):
        """
        @description: The total time for all online sessions.
        @requirement: VideoID, Date (datetime object), EventType
        """
        sessions = getSessions(udata, maxSessionLength=120, minNoActions=3)
        return np.sum(sessions['Duration'])

    def avgSessionTime(self, udata):
        """
        @description: The mean of online session time.
        @requirement: VideoID, Date (datetime object), EventType
        """
        sessions = getSessions(udata, maxSessionLength=120, minNoActions=3)
        return np.mean(sessions['Duration'])

    def stdSessionTime(self, udata):
        """
        @description: The standard deviation of online session time.
        @requirement: VideoID, Date (datetime object), EventType
        """
        sessions = getSessions(udata, maxSessionLength=120, minNoActions=3)
        return np.std(sessions['Duration'])

    def totalClicksWeekdays(self, udata):
        """
        @description: The number of clicks during weekdays.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return len(udata[udata['Weekday'] == 1].index)

    def totalClicksWeekends(self, udata):
        """
        @description: The number of clicks during weekends.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return len(udata[udata['Weekday'] == 0].index)

    def ratioClicksWeekdaysWeekends(self, udata):
        """
        @description: The ratio of weekend to weekday clicks
        @requirement: VideoID, Date (datetime object), EventType
        """
        return self.totalClicksWeekdays(udata) / (self.totalClicksWeekends(udata) + sys.float_info.epsilon)

    def totalClicksOnProblems(self, udata):
        """
        @description: The number of clicks for module “Quiz”.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return len(udata[udata['EventType'].str.contains('Problem.')].index)

    def totalTimeOnProblems(self, udata):
        """
        @description: the total time on module “Quiz”.
        @requirement: VideoID, Date (datetime object), EventType
        """
        udata['PrevEvent'] = udata['EventType'].shift(1)
        udata['PrevProblemID'] = udata['ProblemID'].shift(1)
        udata['TimeDiff'] = udata.TimeStamp.diff()
        udata = udata[(udata['PrevEvent'].str.contains('Problem.')) & (udata['ProblemID'] == udata['PrevProblemID'])]
        return np.sum(udata['TimeDiff'])

    def stdTimeOnProblems(self, udata):
        """
        @description: The standard deviation of time on module “Quiz”.
        @requirement: VideoID, Date (datetime object), EventType
        """
        udata['PrevEvent'] = udata['EventType'].shift(1)
        udata['PrevProblemID'] = udata['ProblemID'].shift(1)
        udata['TimeDiff'] = udata.TimeStamp.diff()
        udata = udata[(udata['PrevEvent'].str.contains('Problem.')) & (udata['ProblemID'] == udata['PrevProblemID'])]
        return np.std(udata['TimeDiff'])