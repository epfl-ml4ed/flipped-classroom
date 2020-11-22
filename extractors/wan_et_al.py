#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractors.extractor import Extractor
import numpy as np

'''
Wan, H., Liu, K., Yu, Q., & Gao, X. (2019). Pedagogical Intervention Practices: Improving Learning Engagement Based on Early Prediction.
In IEEE Transactions on Learning Technologies, 12(2), 278-289.
'''

class WanEtAl(Extractor):

    def __init__(self):
        super().__init__('wan_et_al')

    def getNbFeatures(self):
        """Returns the number of features"""
        return 22

    def getUserFeatures(self, udata):

        features = [
            self.totalDuration(udata),
            self.numberForumPosts(udata),
            self.numberForumPosts(udata),
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
            self.numberForumResponses(udata),
            self.avgNumberSubmissionsPercentile(udata),
            self.avgNumberSubmissionsPercent(udata),
            self.avgStartSubmissionTime(udata),
            self.numberAccessAfterIncorrectSubmission(udata),
            self.totalVideoTimeAfterIncorrectSubmission(udata),
            self.avgTimeBetweenProblemSubmissions(udata),
            self.avgTimeTillFirstCheck(udata),
            self.discussionDurationAfterIncorrectSubmission(udata)
        ]

        if len(features) != self.getNbFeatures():
            raise Exception("getNbFeatures is not up-to-date: {len(features)} != {self.getNbFeatures()}")

        return list(np.nan_to_num(features))

    def totalDuration(self, udata):
        """
        @description: The total time spent on all the resources.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def numberForumPosts(self, udata):
        """
        @description: The number of forum posts posted.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def numberForumBrowse(self, udata):
        """
        @description: The number of forum posts browsed.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def numberSubmissions(self, udata):
        """
        @description: The total number of submissions.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def numberDistinctProblemSubmitted(self, udata):
        """
        @description: The number of distinct problems attempted.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def numberDistinctProblemSubmittedCorrect(self, udata):
        """
        @description:  The number of distinct problems attempted and corrected.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def avgNumberSubmissions(self, udata):
        """
        @description: The average number of submissions per problem.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def observedEventDurationPerCorrectProblem(self, udata):
        """
        @description: The total time spent divided by the number of correct problems.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def submissionsPerCorrectProblem(self, udata):
        """
        @description: TThe number of submissions divided by the number of correct problems.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def avgTimeSolveProblem(self, udata):
        """
        @description: The average time from the first submission to the last submission of each problem.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def observedEventVariance(self, udata):
        """
        @description: The variance of a student's observed event timestamps.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def maxObservedEventDuration(self, udata):
        """
        @description: The maximum duration of observed events.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def totalVideoDuration(self, udata):
        """
        @description: The total time spent on video resources.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def totalPlatformAccess(self, udata):
        """
        @description: TThe number of platform access / logins.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def numberForumResponses(self, udata):
        """
        @description: The number of forum responses.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

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
        return

    def correctSubmissionsPercent(self, udata):
        """
        @description: The percentage of the total submissions that were correct.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def avgStartSubmissionTime(self, udata):
        """
        @description: The average first submitting time after the problem released.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def numberAccessAfterIncorrectSubmission(self, udata):
        """
        @description: The number of platform access / logins after the problem submitted incorrect.
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
        return

    def avgTimeTillFirstCheck(self, udata):
        """
        @description: The average time between the problem first chec and the problem first get of all problems.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def discussionDurationAfterIncorrectSubmission(self, udata):
        """
        @description: The total discussion duration after incorrect submission.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return
