#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractors.extractor import Extractor
from helpers.db_query import *

import numpy as np

'''
Wan, H., Liu, K., Yu, Q., & Gao, X. (2019). Pedagogical Intervention Practices: Improving Learning Engagement Based on Early Prediction.
In IEEE Transactions on Learning Technologies, 12(2), 278-289.
'''

class WanEtAl(Extractor):

    def __init__(self, name='base'):
        """
        @description: Returns the identifier associated with this feature set.
        """
        super().__init__('wan_et_al')

    def getNbFeatures(self):
        """
        @description: Returns the number of expected features.
        """
        return 17

    def getUserFeatures(self, udata, wid, year):
        """
        @description: Returns the user features computed from the udata
        """

        udata = udata.copy()
        udata['TimeStamp'] = udata['Date']
        udata = udata.sort_values(by='TimeStamp')

        features = [
            self.totalDuration(udata),
            self.numberSubmissions(udata),
            self.numberDistinctProblemSubmitted(udata),
            self.numberDistinctProblemSubmittedCorrect(udata),
            self.avgNumberSubmissions(udata),
            self.observedEventDurationPerCorrectProblem(udata),
            self.submissionsPerCorrectProblem(udata),
            self.avgTimeSolveProblem(udata),
            self.observedEventVariance(udata),
            self.maxObservedEventDuration(udata),
            self.totalVideoDuration(udata),
            self.totalPlatformAccess(udata),
            self.avgNumberSubmissionsPercentile(udata),
            self.avgNumberSubmissionsPercent(udata),
            self.avgStartSubmissionTime(udata),
            self.totalVideoTimeAfterIncorrectSubmission(udata),
            self.avgTimeBetweenProblemSubmissions(udata)
        ]

        if len(features) != self.getNbFeatures():
            raise Exception("getNbFeatures is not up-to-date: {len(features)} != {self.getNbFeatures()}")

        return list(np.nan_to_num(features))

    def totalDuration(self, udata):
        """
        @description: The total time spent on all the resources.
        @requirement: VideoID, Date (datetime object), EventType
        """
        sessions = getSessions(udata, maxSessionLength=120, minNoActions=3)
        return np.sum(sessions['Duration'])

    def numberSubmissions(self, udata):
        """
        @description: The total number of submissions.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return len(udata[udata['EventType'].str.contains('Problem.Check')])

    def numberDistinctProblemSubmitted(self, udata):
        """
        @description: The number of distinct problems attempted.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return len(udata[udata['EventType'].str.contains('Problem.Check')]['ProblemID'].unique())

    def numberDistinctProblemSubmittedCorrect(self, udata):
        """
        @description:  The number of distinct problems attempted and corrected.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return len(udata[(udata['EventType'].str.contains('Problem.Check')) & (udata['Grade'] == 100.0)]['ProblemID'].unique())

    def avgNumberSubmissions(self, udata):
        """
        @description: The average number of submissions per problem.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return np.mean(udata[udata['EventType'].str.contains('Problem.Check')].groupby(by='ProblemID').size())

    def observedEventDurationPerCorrectProblem(self, udata):
        """
        @description: The total time spent divided by the number of correct problems.
        @requirement: VideoID, Date (datetime object), EventType
        """
        correctProblems = udata[(udata['EventType'].str.contains('Problem.Check')) & (udata['Grade'] == 100.0)]['ProblemID'].unique()
        udata = udata[udata['ProblemID'].isin(correctProblems)]
        timeSolving = []
        for index, group in udata.groupby(by='ProblemID'):
            group['TimeDiff'] = udata.TimeStamp.diff()
            timeSolving += ([] if len(group.index) < 2 else [np.sum([s for s in group['TimeDiff'] if s >= 0 and s < 120 * 60])])
        return np.mean(timeSolving)

    def submissionsPerCorrectProblem(self, udata):
        """
        @description: The number of submissions divided by the number of correct problems.
        @requirement: VideoID, Date (datetime object), EventType
        """
        correctProblems = udata[(udata['EventType'].str.contains('Problem.Check')) & (udata['Grade'] == 100.0)]['ProblemID'].unique()
        return len(udata[(udata['EventType'].str.contains('Problem.Check')) & (udata['ProblemID'].isin(correctProblems))]) / len(correctProblems)

    def avgTimeSolveProblem(self, udata):
        """
        @description: The average time from the first submission to the last submission of each problem.
        @requirement: VideoID, Date (datetime object), EventType
        """
        timeSolving = []
        for index, group in udata.groupby(by='ProblemID'):
            timeSolving += ([] if len(group.index) < 2 else [group['TimeStamp'].tolist()[-1] - group['TimeStamp'].tolist()[0]])
        return np.mean(timeSolving)

    def observedEventVariance(self, udata):
        """
        @description: The variance of a student's observed event timestamps.
        @requirement: VideoID, Date (datetime object), EventType
        """
        udata['PrevEvent'] = udata['EventType'].shift(1)
        udata['PrevProblemID'] = udata['ProblemID'].shift(1)
        udata['TimeDiff'] = udata.TimeStamp.diff()
        return np.var(udata['TimeDiff'])

    def maxObservedEventDuration(self, udata):
        """
        @description: The maximum duration of observed events.
        @requirement: VideoID, Date (datetime object), EventType
        """
        udata['PrevEvent'] = udata['EventType'].shift(1)
        udata['PrevProblemID'] = udata['ProblemID'].shift(1)
        udata['TimeDiff'] = udata.TimeStamp.diff()
        return np.max(udata['TimeDiff'])

    def totalVideoDuration(self, udata):
        """
        @description: The total time spent on video resources.
        @requirement: VideoID, Date (datetime object), EventType
        """
        sessions = getSessions(udata[udata['EventType'].str.contains('Video.')], maxSessionLength=120, minNoActions=3)
        return np.sum(sessions['Duration'])

    def totalPlatformAccess(self, udata):
        """
        @description: The number of platform access / logins.
        @requirement: VideoID, Date (datetime object), EventType
        """
        sessions = getSessions(udata, maxSessionLength=120, minNoActions=3)
        return len(sessions.index)

    def avgNumberSubmissionsPercentile(self, udata):
        """
        @description: The student's submission count divided by the average of all the students' submissions count.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def avgNumberSubmissionsPercent(self, udata):
        """
        @description: The student's submission count divided by the maximum of all the students' submissions count.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def numberSubmissionsFinalCorrectProblems(self, udata):
        """
        @description: The number of submissions for final correct problems.
        @requirement: VideoID, Date (datetime object), EventType
        """
        correctProblems = udata[(udata['EventType'].str.contains('Problem.Check')) & (udata['Grade'] == 100.0)]['ProblemID'].unique()
        return len(udata[(udata['EventType'].str.contains('Problem.Check')) & (udata['ProblemID'].isin(correctProblems))])

    def correctSubmissionsPercent(self, udata):
        """
        @description: The percentage of the total submissions that were correct.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return len(udata[(udata['EventType'].str.contains('Problem.Check')) & (udata['Grade'] == 100.0)]) / len(udata[(udata['EventType'].str.contains('Problem.Check')) & (udata['Grade'].notnull())])

    def avgStartSubmissionTime(self, udata):
        """
        @description: The average first submitting time after the problem released.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def totalVideoTimeAfterIncorrectSubmission(self, udata):
        """
        @description: The video watching time after the problem submit incorrect.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def avgTimeBetweenProblemSubmissions(self, udata):
        """
        @description: The average duration between problem submissions.
        @requirement: VideoID, Date (datetime object), EventType
        """
        udata = udata[udata['EventType'].str.contains('Problem.Check') & (udata['Grade'].notnull())]
        udata['PrevEvent'] = udata['EventType'].shift(1)
        udata['PrevProblemID'] = udata['ProblemID'].shift(1)
        udata['TimeDiff'] = udata.TimeStamp.diff()
        udata = udata[(udata['TimeDiff'] > 0.0) & (udata['ProblemID'] == udata['PrevProblemID'])]
        return np.mean(udata['TimeDiff'])
