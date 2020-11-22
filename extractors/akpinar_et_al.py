#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractors.extractor import Extractor
from helpers.time import string2Datetime
from helpers.db_query import *

from datetime import datetime as dt
from itertools import combinations
import numpy as np

'''
Akpinar, N. J., Ramdas, A., & Acar, U. (2020). Analyzing Student Strategies In Blended Courses Using Clickstream Data.
In Thirteenth International Conference on Educational Data Mining (EDM 2020).
'''

class AkpinarEtAl(Extractor):

    def __init__(self, name='base'):
        """
        @description: Returns the identifier associated with this feature set.
        """
        super().__init__('akpinar_et_al')
        self.utypes = list(np.unique(getVideoEventsInfo()['EventType']))

    def getNbFeatures(self):
        """
        @description: Returns the number of expected features.
        """
        return 12

    def getUserFeatures(self, udata, wid, year, ngram=3):
        """
        @description: Returns the user features computed from the udata
        """

        features = [self.numberSessions(udata), self.totalClicks(udata), self.attendanceVideos(udata), self.attendanceProblems(udata)] + self.countNGrams(udata, ngram)

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
        tmpudata = udata.copy()
        tmpudata['TimeStamp'] = tmpudata['TimeStamp'].apply(lambda x: string2Datetime(dt.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S')))
        return len(getSessions(tmpudata, maxSessionLength=120, minNoActions=3).index)

    def attendanceVideos(self, udata):
        """
        @description: The time spent in watching videos.
        @requirement: VideoID, Date (datetime object), EventType
        """
        tmpudata = udata.copy().sort_values(by='TimeStamp')
        tmpudata['PrevEvent'] = tmpudata['EventType'].shift(1)
        tmpudata['PrevVideoID'] = tmpudata['VideoID'].shift(1)
        tmpudata['TimeDiff'] = tmpudata.TimeStamp.diff().dropna()
        tmpudata = tmpudata[(tmpudata['PrevEvent'].str.contains('Video.')) & (tmpudata['VideoID'] == tmpudata['PrevVideoID'])]
        return np.sum(tmpudata['TimeDiff'])

    def attendanceProblems(self, udata):
        """
        @description: The time spent in playing with problems.
        @requirement: VideoID, Date (datetime object), EventType
        """
        tmpudata = udata.copy().sort_values(by='TimeStamp')
        tmpudata['PrevEvent'] = tmpudata['EventType'].shift(1)
        tmpudata['PrevProblemID'] = tmpudata['ProblemID'].shift(1)
        tmpudata['TimeDiff'] = tmpudata.TimeStamp.diff().dropna()
        tmpudata = tmpudata[(tmpudata['PrevEvent'].str.contains('Problem.')) & (tmpudata['ProblemID'] == tmpudata['PrevProblemID'])]
        return np.sum(tmpudata['TimeDiff'])

    def countNGrams(self, udata, ngram=3):
        """
        @description: The time spent in playing with problems.
        @requirement: VideoID, Date (datetime object), EventType
        """
        tmpudata = udata.copy()
        tmpudata['TimeStamp'] = tmpudata['TimeStamp'].apply(lambda x: string2Datetime(dt.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S')))
        sessions = getSessions(tmpudata, maxSessionLength=120, minNoActions=3)

        combs = [comb for comb in combinations(self.utypes, ngram)]
        maps = {c : i for i, c in enumerate(combs)}
        counts = np.zeros(len(combs))

        for s in sessions['Event']:
            if len(s) >= ngram:
               for i in range(len(s) - ngram + 1):
                   c = (s[i],)
                   for j in range(i+1, i + ngram):
                       c += (s[j],)
                   counts[maps[c]] += 1

        return list(counts)
