#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractors.extractor import Extractor
from helpers.db_query import *

from itertools import product
import numpy as np

'''
Akpinar, N. J., Ramdas, A., & Acar, U. (2020). Analyzing Student Strategies In Blended Courses Using Clickstream Data.
In Thirteenth International Conference on Educational Data Mining (EDM 2020).
'''

class AkpinarEtAl(Extractor):

    def __init__(self, name='base', ngram=3):
        """
        @description: Returns the identifier associated with this feature set.
        """
        super().__init__('akpinar_et_al')
        self.ngram = ngram
        utypes = list(np.unique(getVideoEventsInfo()['EventType']))
        utypes += list(np.unique(getProblemEventsInfo()['EventType']))
        self.perms = list(product(utypes, repeat=ngram))

    def getNbFeatures(self):
        """
        @description: Returns the number of expected features.
        """
        return 4 + len(self.perms)

    def getUserFeatures(self, udata, wid, year):
        """
        @description: Returns the user features computed from the udata
        """

        features = [self.numberSessions(udata), self.totalClicks(udata), self.attendanceVideos(udata), self.attendanceProblems(udata)] + self.countNGrams(udata)

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
        udata = udata.copy()
        udata['TimeStamp'] = udata['Date']
        udata = udata.sort_values(by='TimeStamp')
        return len(getSessions(udata, maxSessionLength=120, minNoActions=3).index)

    def attendanceVideos(self, udata):
        """
        @description: The time spent in watching videos.
        @requirement: VideoID, Date (datetime object), EventType
        """
        udata = udata.copy()
        udata['TimeStamp'] = udata['Date']
        udata = udata.sort_values(by='TimeStamp')
        udata['PrevEvent'] = udata['EventType'].shift(1)
        udata['PrevVideoID'] = udata['VideoID'].shift(1)
        udata['TimeDiff'] = udata.TimeStamp.diff()
        udata['TimeDiff'] = udata['TimeDiff'].apply(lambda x : x.total_seconds())
        udata = udata[(udata['PrevEvent'].str.contains('Video.')) & (udata['VideoID'] == udata['PrevVideoID'])]
        return np.sum(udata['TimeDiff'])

    def attendanceProblems(self, udata):
        """
        @description: The time spent in playing with problems.
        @requirement: VideoID, Date (datetime object), EventType
        """
        udata = udata.copy()
        udata['TimeStamp'] = udata['Date']
        udata = udata.sort_values(by='TimeStamp')
        udata['PrevEvent'] = udata['EventType'].shift(1)
        udata['PrevProblemID'] = udata['ProblemID'].shift(1)
        udata['TimeDiff'] = udata.TimeStamp.diff()
        udata['TimeDiff'] = udata['TimeDiff'].apply(lambda x : x.total_seconds())
        udata = udata[(udata['PrevEvent'].str.contains('Problem.')) & (udata['ProblemID'] == udata['PrevProblemID'])]
        return np.sum(udata['TimeDiff'])

    def countNGrams(self, udata):
        """
        @description: The time spent in playing with problems.
        @requirement: VideoID, Date (datetime object), EventType
        """
        udata = udata.copy()
        udata['TimeStamp'] = udata['Date']
        udata = udata.sort_values(by='TimeStamp')
        sessions = getSessions(udata, maxSessionLength=120, minNoActions=3)

        maps = {c : i for i, c in enumerate(self.perms)}
        counts = np.zeros(len(self.perms))

        for events in sessions['Event']:
            s = events.split(',')
            if len(s) >= self.ngram:
                for i in range(len(s) - self.ngram + 1):
                    c = (s[i],)
                    for j in range(i+1, i + self.ngram):
                        c += (s[j],)
                    counts[maps[c]] += 1

        return list(counts)
