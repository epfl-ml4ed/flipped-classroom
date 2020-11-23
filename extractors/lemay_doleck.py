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
        """
        @description: Returns the identifier associated with this feature set.
        """
        super().__init__('lemay_doleck')

    def getNbFeatures(self):
        """
        @description: Returns the number of expected features.
        """
        return 10

    def getUserFeatures(self, udata, wid, year):
        """
        @description: Returns the user features computed from the udata
        """

        udata = udata.copy()
        udata['TimeStamp'] = udata['Date']
        udata = udata.sort_values(by='TimeStamp')

        features = [
            self.fracSpent(udata, wid, year) if len(udata) > 1 else 0,
            self.fracComp(udata, wid, year) if len(udata) > 1 else 0,
            self.fracPlayed(udata, wid, year) if len(udata) > 1 else 0,
            self.fracPaused(udata, wid, year) if len(udata) > 1 else 0,
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

    def fracSpent(self, udata, wid, year):
        """
        @description: The fraction of (real) time the user spent playing the video, relative to its length.
        @requirement: VideoID, Date (datetime object), EventType
        """
        udata['PrevEvent'] = udata['EventType'].shift(1)
        udata['PrevVideoID'] = udata['VideoID'].shift(1)
        udata['TimeDiff'] = udata.TimeStamp.diff()
        udata['TimeDiff'] = udata['TimeDiff'].apply(lambda x : x.total_seconds())
        udata = udata[(udata['PrevEvent'].str.contains('Video.')) & (udata['VideoID'] == udata['PrevVideoID'])]

        course_schedule = get_dated_videos()
        taught_schedule = course_schedule[(course_schedule['Year'] == year) & (course_schedule['Week'] <= wid)]

        maps = {v: d for v, d in zip(udata.groupby(by='VideoID').sum().index, udata.groupby(by='VideoID').sum()['TimeDiff'])}

        ratio = []
        for videoID, timeSchedule in zip(taught_schedule['VideoID'], taught_schedule['Duration']):
            ratio.append((maps[videoID] if videoID in maps else 0) / timeSchedule)

        return np.mean(ratio)

    def fracComp(self, udata, wid, year):
        """
        @description: The percentage of the video that the user played, not counting repeated play position intervals; hence, it must be between 0 and 1.
        @requirement: VideoID, Date (datetime object), EventType
        """
        udata = udata.sort_values(by=['VideoID', 'CurrentTime'])
        udata['PrevEvent'] = udata['EventType'].shift(1)
        udata['PrevVideoID'] = udata['VideoID'].shift(1)
        udata['TimeDiff'] = udata.TimeStamp.diff()
        udata = udata[(udata['PrevEvent'].str.contains('Video.Play')) & (udata['VideoID'] == udata['PrevVideoID'])]
        udata = udata.drop_duplicates(subset=['VideoID'], keep='last')

        course_schedule = get_dated_videos()
        taught_schedule = course_schedule[(course_schedule['Year'] == year) & (course_schedule['Week'] <= wid)]

        maps = {v: d for v, d in zip(udata.groupby(by='VideoID').sum().index, udata.groupby(by='VideoID').sum()['CurrentTime'])}

        ratio = []
        for videoID, timeSchedule in zip(taught_schedule['VideoID'], taught_schedule['Duration']):
            ratio.append((maps[videoID] if videoID in maps else 0) / timeSchedule)

        return np.mean(ratio)

    def fracPlayed(self, udata, wid, year):
        """
        @description: The amount of the video that the user played, with repetition, relative to its length.
        @requirement: VideoID, Date (datetime object), EventType
        """
        udata['PrevEvent'] = udata['EventType'].shift(1)
        udata['PrevVideoID'] = udata['VideoID'].shift(1)
        udata['TimeDiff'] = udata.TimeStamp.diff()
        udata['TimeDiff'] = udata['TimeDiff'].apply(lambda x : x.total_seconds())
        udata = udata[(udata['PrevEvent'].str.contains('Video.Play')) & (udata['VideoID'] == udata['PrevVideoID'])]

        course_schedule = get_dated_videos()
        taught_schedule = course_schedule[(course_schedule['Year'] == year) & (course_schedule['Week'] <= wid)]

        maps = {v: d for v, d in zip(udata.groupby(by='VideoID').sum().index, udata.groupby(by='VideoID').sum()['TimeDiff'])}

        ratio = []
        for videoID, timeSchedule in zip(taught_schedule['VideoID'], taught_schedule['Duration']):
            ratio.append((maps[videoID] if videoID in maps else 0) / timeSchedule)

        return np.mean(ratio)

    def fracPaused(self, udata, wid, year):
        """
        @description: The fraction of time the user spent paused on the video, relative to its length.
        @requirement: VideoID, Date (datetime object), EventType
        """
        udata['PrevEvent'] = udata['EventType'].shift(1)
        udata['PrevVideoID'] = udata['VideoID'].shift(1)
        udata['TimeDiff'] = udata.TimeStamp.diff()
        udata['TimeDiff'] = udata['TimeDiff'].apply(lambda x : x.total_seconds())
        udata = udata[(udata['PrevEvent'].str.contains('Video.Pause')) & (udata['VideoID'] == udata['PrevVideoID'])]

        course_schedule = get_dated_videos()
        taught_schedule = course_schedule[(course_schedule['Year'] == year) & (course_schedule['Week'] <= wid)]

        maps = {v: d for v, d in zip(udata.groupby(by='VideoID').sum().index, udata.groupby(by='VideoID').sum()['TimeDiff'])}

        ratio = []
        for videoID, timeSchedule in zip(taught_schedule['VideoID'], taught_schedule['Duration']):
            ratio.append((maps[videoID] if videoID in maps else 0) / timeSchedule)

        return np.mean(ratio)

    def avgPlayBackRate(self, udata):
        """
        @description: The time-average of the playback rates selected by the user. The player on Coursera allows rates between 0.75x and 2.0x the default speed.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return np.mean(udata[udata['EventType'].str.contains('Video.SpeedChange')]['NewSpeed'])

    def stdPlayBackRate(self, udata):
        """
        @description: The standard deviation of the playback rates selected over time.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return np.std(udata[udata['EventType'].str.contains('Video.SpeedChange')]['NewSpeed'])

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
        return count_actions(udata, 'Video.SeekForward')

    def totalVideoWatcherPerWeek(self, udata):
        """
        @description: The number of videos viewed per week.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return np.mean(udata[udata['EventType'].str.contains('Video.')][['VideoID', 'Week']].drop_duplicates().groupby(by='Week').size())