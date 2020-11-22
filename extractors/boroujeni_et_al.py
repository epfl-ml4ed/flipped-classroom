#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractors.extractor import Extractor
from helpers.feature_extraction import *

import numpy as np

'''
Boroujeni, M. S., Sharma, K., Kidziński, Ł., Lucignano, L., & Dillenbourg, P. (2016, September). How to quantify student’s regularity?
In European Conference on Technology Enhanced Learning (pp. 277-291). Springer, Cham.
'''

class BoroujeniEtAl(Extractor):

    def __init__(self):
        super().__init__('boroujeni_et_al')

    def getNbFeatures(self):
        """Returns the number of features"""
        return 12

    def getUserFeatures(self, udata):

        videoEvents = udata[udata.EventType.str.contains('Video')]
        problemEvents = udata[udata.EventType.str.contains('Problem')]

        features = [
            self.peakDayHour(udata),
            self.peakWeekDay(udata),
            self.weeklySimilarity1(udata),
            self.weeklySimilarity2(udata),
            self.weeklySimilarity3(udata),
            self.freqDayHour(udata),
            self.freqWeekDay(udata),
            self.freqWeekHour(udata),
            self.nbQuiz(problemEvents),
            self.propQuiz(problemEvents),
            self.intervalVideoQuiz(videoEvents, problemEvents),
            self.semesterRepartitionQuiz(problemEvents)
        ]

        if len(features) != self.getNbFeatures():
            raise Exception("getNbFeatures is not up-to-date: {len(features)} != {self.getNbFeatures()}")

        return list(np.nan_to_num(features))

    def peakDayHour(self, udata):
        """
        @description: Identifies if user’s activities are concentrated around a particular hour of the day.
        @requirement: TimeStamp
        """
        return PDH(udata)

    def peakWeekDay(self, udata):
        """
        @description: Identifies if user’s activities are concentrated around a particular day of the week.
        @requirement: TimeStamp
        """
        return PWD(udata)

    def weeklySimilarity1(self, udata):
        """
        @description: Measures if the user works on the same weekdays throughout the weeks.
        @requirement: TimeStamp
        """
        return WS1(udata)

    def weeklySimilarity2(self, udata):
        """
        @description: Compares the normalized profiles and measure if the user has a similar distribution of workload among weekdays, in different weeks of the course.
        @requirement: TimeStamp
        """
        return WS2(udata)

    def weeklySimilarity3(self, udata):
        """
        @description: Compares the original profiles and reflects if the time spent on each day of the week is similar for different weeks of the course.
        @requirement: TimeStamp
        """
        return WS3(udata)

    def freqDayHour(self, udata):
        """
        @description: Evaluates the intensity of a daily period, i.e., if the user works periodically at a specific hour of the day. The value is the Fourier transform of the active days (day with at least one event) evaluated at the frequency 1 / nb of hours in a day = 1 / 24.
        @requirement: TimeStamp
        """
        return FDH(udata)

    def freqWeekDay(self, udata):
        """
        @description: Evaluates the intensity of a weekly period, i.e., if the user works periodically on a specific day of the week. The value is the Fourier transform of the active days (day with at least one event) evaluated at the frequency 1 / nb of days in a week = 1 / 7
        @requirement: TimeStamp
        """
        return FWD(udata)

    def freqWeekHour(self, udata):
        """
        @description: Evaluates the intensity of the period 7*24, i.e., if the user works periodically at a specific hour of the week. The value is the Fourier transform of the active days (day with at least one event) evaluated at the frequency 1 / nb of hours in a week = 1 / (7*24)
        @requirement: TimeStamp
        """
        return FWH(udata)

    def nbQuiz(self, udata):
        """
        @description: Total count of quiz completed over the whole semester
        @requirement: ProblemID
        """
        return NQZ(udata)

    def propQuiz(self, udata):
        """
        @description: Proportion of quiz completed over the flipped period
        @requirement: Year
        """
        return PQZ(udata)

    def intervalVideoQuiz(self, video_events, problem_events):
        """
        @description: For every completed quiz, compute the time intervals (minutes) between the first viewing of the video and the quiz completion and return the interquartile range of the time intervals video_events
        @requirement: Videos (VideoID, EventType, TimeStamp, Subchapter) & Problems (ProblemID, EventType, TimeStamp, Subchapter)
        """
        return IVQ(video_events, problem_events)

    def semesterRepartitionQuiz(self, udata):
        """
        @description: Measures the repartition of the quiz completions. The std (in hours) of the time intervals is computed aswell as the dates of completions. The smaller the std is, the more regular the student is.
        @requirement: AccountUserID, ProblemID, EventType, TimeStamp
        """
        return SRQ(udata)