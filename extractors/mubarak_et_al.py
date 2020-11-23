#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractors.extractor import Extractor
from helpers.feature_extraction import *

import numpy as np

'''
Mubarak, A. A., Cao, H., & Ahmed, S. A. (2020). Predictive learning analytics using deep learning model in MOOCs’ courses videos.
In Education and Information Technologies, 1-22.
'''

class MubarakEtAl(Extractor):

    def __init__(self, name='base'):
        """
        @description: Returns the identifier associated with this feature set.
        """
        super().__init__('mubarak_et_al')

    def getNbFeatures(self):
        """
        @description: Returns the number of expected features.
        """
        return 13

    def getUserFeatures(self, udata, wid, year):
        """
        @description: Returns the user features computed from the udata
        """

        udata = udata.copy()
        udata['TimeStamp'] = udata['Date']
        udata = udata.sort_values(by='TimeStamp')

        features = [
            self.fracComp(udata, wid, year),
            self.fracSpent(udata, wid, year),
            self.fracPL(udata, wid, year),
            self.numPL(udata),
            self.numPa(udata),
            self.fracPa(udata, wid, year),
            self.fracFrwd(udata),
            self.fracBkwd(udata),
            self.numSBW(udata),
            self.numFFW(udata),
            self.avChR(udata),
            self.numLD(udata),
            self.numCompt(udata, wid, year),
        ]

        if len(features) != self.getNbFeatures():
            raise Exception("getNbFeatures is not up-to-date: {len(features)} != {self.getNbFeatures()}")

        return list(np.nan_to_num(features))

    def fracComp(self, udata, wid, year):
        """
        @description: The percentage of the video which the learner watched, not counting repeated segment intervals. The completed fraction is a tentative measure of how closely learners are aligned with a video. So, values have to be like [0,1].
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

    def fracSpent(self, udata, wid, year):
        """
        @description: The amount of (real) time the learner spent watching the video (i.e. when playing or pausing), compared to the video’s actual duration. FracSpent often reflects the amount of time that it took for the learner to grasp the given content. It can take more than 1 value. And is thus a reflection of the consistency and complexity of the delivery of the material. The high value fracSpent may mean that the l.
        @requirement: VideoID, Date (datetime object), EventType
        """
        udata['PrevEvent'] = udata['EventType'].shift(1)
        udata['PrevVideoID'] = udata['VideoID'].shift(1)
        udata['TimeDiff'] = udata.TimeStamp.diff()
        udata = udata[(udata['PrevEvent'].str.contains('Video.')) & (udata['VideoID'] == udata['PrevVideoID'])]

        course_schedule = get_dated_videos()
        taught_schedule = course_schedule[(course_schedule['Year'] == year) & (course_schedule['Week'] <= wid)]

        maps = {v: d for v, d in zip(udata.groupby(by='VideoID').sum().index, udata.groupby(by='VideoID').sum()['TimeDiff'])}

        ratio = []
        for videoID, timeSchedule in zip(taught_schedule['VideoID'], taught_schedule['Duration']):
            ratio.append((maps[videoID] if videoID in maps else 0) / timeSchedule)

        return np.mean(ratio)

    def fracPL(self, udata, wid, year):
        """
        @description: The cumulative amount of play time that the learner was watching, divided by a video’s real duration. Playback periods of under 5 s have been deleted. The fracPL measures repetitive fragments that are watched as opposed to fracComp, which can thus take value greater than 1.
        @requirement: VideoID, Date (datetime object), EventType
        """
        udata['PrevEvent'] = udata['EventType'].shift(1)
        udata['PrevVideoID'] = udata['VideoID'].shift(1)
        udata['TimeDiff'] = udata.TimeStamp.diff()
        udata = udata[(udata['PrevEvent'].str.contains('Video.Play')) & (udata['VideoID'] == udata['PrevVideoID'])]

        course_schedule = get_dated_videos()
        taught_schedule = course_schedule[(course_schedule['Year'] == year) & (course_schedule['Week'] <= wid)]

        maps = {v: d for v, d in zip(udata.groupby(by='VideoID').sum().index, udata.groupby(by='VideoID').sum()['TimeDiff'])}

        ratio = []
        for videoID, timeSchedule in zip(taught_schedule['VideoID'], taught_schedule['Duration']):
            ratio.append((maps[videoID] if videoID in maps else 0) / timeSchedule)

        return np.mean(ratio)

    def numPL(self, udata):
        """
        @description: The number of times that video was played by the learner. This function may indicate the learner’s expended efforts to internalize the material further. An abnormally high-value NumPL can therefore suggest that the video quality is unclear, or that the quality is incredibly difficult.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return count_actions(udata, 'Video.Play')

    def numPa(self, udata):
        """
        @description: The amount of times the learner has tapped on the pause case in the video. Similar to NumPL, a high value of this attribute suggests the extra effort has been made by the learner to grasp the video material or to perform an off-task conduct.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return count_actions(udata, 'Video.Pause')

    def fracPa(self, udata, wid, year):
        """
        @description: The real-time fraction the learner spent pausing on the video, divided on its total playback time. It is possible for fracPa to get up value >1. This feature could take larger values, so that it indicates that the learner exhibited extra effort to understand the material
        @requirement: VideoID, Date (datetime object), EventType
        """
        udata['PrevEvent'] = udata['EventType'].shift(1)
        udata['PrevVideoID'] = udata['VideoID'].shift(1)
        udata['TimeDiff'] = udata.TimeStamp.diff()
        udata = udata[(udata['PrevEvent'].str.contains('Video.Pause')) & (udata['VideoID'] == udata['PrevVideoID'])]

        course_schedule = get_dated_videos()
        taught_schedule = course_schedule[(course_schedule['Year'] == year) & (course_schedule['Week'] <= wid)]

        maps = {v: d for v, d in zip(udata.groupby(by='VideoID').sum().index, udata.groupby(by='VideoID').sum()['TimeDiff'])}

        ratio = []
        for videoID, timeSchedule in zip(taught_schedule['VideoID'], taught_schedule['Duration']):
            ratio.append((maps[videoID] if videoID in maps else 0) / timeSchedule)

        return np.mean(ratio)

    def fracFrwd(self, udata):
        """
        @description: The sum of the video the learner had to skip forward while the video was running or pausing; divided on its duration, with repetition. It needs values from 0 to 1.
        @requirement: VideoID, Date (datetime object), EventType
        """
        udata = udata[(udata['EventType'] == 'Video.Seek') & (udata['OldTime'] < udata['NewTime'])]
        return np.mean(udata['NewTime'] - udata['OldTime'])

    def fracBkwd(self, udata):
        """
        @description: The sum of time the learner had to skip back when the time was playing or pausing, divided by its duration, with repetition. It takes values from 0 to 1 This feature can take values greater than 1 and may mean that the learner has expended extra time internalizing the video passages. It also indicates that the subject is more interested in or confided in these passages.
        @requirement: VideoID, Date (datetime object), EventType
        """
        udata = udata[(udata['EventType'] == 'Video.Seek') & (udata['OldTime'] > udata['NewTime'])]
        return np.mean(udata['OldTime'] - udata['NewTime'])

    def numSBW(self, udata):
        """
        @description: The sum of points in the video the learner was skipping backward. A higher value of this feature indicates the video is interesting or difficult in content.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return count_actions(udata, 'Video.SeekBackward')

    def numFFW(self, udata):
        """
        @description: The number of times in the video the learner has hopped forward. In this event a high value indicates that the learner is aware that the content of this video clip is not important
        @requirement: VideoID, Date (datetime object), EventType
        """
        return count_actions(udata, 'Video.SeekForward')

    def avChR(self, udata):
        """
        @description: Time-average change of playback Rate selected in the play state by the learner. The Stanford MOOC player allows the default speed rates of between 0.50x and 2.0x. Analysis of average change rate may show that learners viewed the video content quickly or slowly.
        @requirement: VideoID, Date (datetime object), EventType
        """
        udata = udata[udata['EventType'] == 'Video.SpeedChange']
        udata['PrevEvent'] = udata['EventType'].shift(1)
        udata['PrevVideoID'] = udata['VideoID'].shift(1)
        udata['TimeDiff'] = udata.TimeStamp.diff()
        udata = udata[udata['VideoID'] == udata['PrevVideoID']]
        return np.mean(udata['TimeDiff'])

    def numLD(self, udata):
        """
        @description: The learner has loaded the video the number of times. A value in this feature indicates the learner can watch the video without having to associate it with the content of the course
        @requirement: VideoID, Date (datetime object), EventType
        """
        return count_actions(udata, 'Video.Load')

    def numCompt(self, udata, wid, year):
        """
        @description: The sum of times a learner viewed a video in its entirety. This feature must have a value less than or equal to the number of video playbacks. It also indicates a learner is interested in viewing the whole video.
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

        counter = 0
        for videoID, timeSchedule in zip(taught_schedule['VideoID'], taught_schedule['Duration']):
            counter += (1 if videoID in maps and abs(maps[videoID] - timeSchedule) < 15.0 else 0)

        return counter