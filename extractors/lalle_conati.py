#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractors.extractor import Extractor
from helpers.feature_extraction import *

import numpy as np

'''
Lallé, S., & Conati, C. (2020, July). A Data-Driven Student Model to Provide Adaptive Support During Video Watching Across MOOCs.
In International Conference on Artificial Intelligence in Education (pp. 282-295). Springer, Cham.
'''

class LalleConati(Extractor):

    def __init__(self, name='base'):
        """
        @description: Returns the identifier associated with this feature set.
        """
        super().__init__('lalle_conati')

    def getFeatureNames(self):
        """
        @description: Returns the names of the feature in the same order as getUserFeatures
        """
        return ["totalViews", "avgWeeklyPropWatched", "stdWeeklyPropWatched", "avgWeeklyPropReplayed", 
                "stdWeeklyPropReplayed", "avgWeeklyPropInterrupted", "stdWeeklyPropInterrupted", 
                "totalActions", "freqAllActions", "freqPlay", "freqPause", "freqSeekBackward",
                "freqSeekForward", "freqSpeedChange", "freqStop", "avgPauseDuration", "stdPauseDuration",
                "avgSeekLength", "stdSeekLength", "avgTimeSpeedingUp", "stdTimeSpeedingUp"]

    def getNbFeatures(self):
        """
        @description: Returns the number of expected features.
        """
        return 21
    
    def getUserFeatures(self, udata, wid, year):
        """
        @description: Returns the user features computed from the udata
        """
        features = [
            self.totalViews(udata),
            self.avgWeeklyPropWatched(udata),
            self.stdWeeklyPropWatched(udata),
            self.avgWeeklyPropReplayed(udata),
            self.stdWeeklyPropReplayed(udata),
            self.avgWeeklyPropInterrupted(udata),
            self.stdWeeklyPropInterrupted(udata),
            self.totalActions(udata),
            self.freqAllActions(udata),
            self.freqPlay(udata),
            self.freqPause(udata),
            self.freqSeekBackward(udata),
            self.freqSeekForward(udata),
            self.freqSpeedChange(udata),
            self.freqStop(udata),
            self.avgPauseDuration(udata),
            self.stdPauseDuration(udata),
            self.avgSeekLength(udata),
            self.stdSeekLength(udata),
            self.avgTimeSpeedingUp(udata),
            self.stdTimeSpeedingUp(udata)
        ]

        if len(features) != self.getNbFeatures():
            raise Exception(f"getNbFeatures is not up-to-date: {len(features)} != {self.getNbFeatures()}")

        return list(np.nan_to_num(features))

    def totalActions(self, udata):
        """
        @description: Counts the total number of actions performed across every videos
        """
        return len(udata)

    def totalViews(self, udata):
        """
        @description: Counts the total of videos views (rewatch and interruption included)
        @assumption: consider that a video is watched at most once per day
        @requirement: VideoID, Date (datetime object)
        """
        copy = udata.copy()
        copy['Day'] = udata.Date.dt.date
        # From the assumption the video view is a unique pair (video id, day)
        return len(copy.drop_duplicates(subset=['VideoID', 'Day']))

    def avgWeeklyPropWatched(self, udata):
        """
        @description: Computes the average proportion of videos watched per week. In other words the number of videos watched (only counting the ones assigned) over the total number of videos assigned.
        @requirement: VideoID, Year (YYYY format), Date (datetime object)
        """
        return weekly_prop_watched(udata).mean()

    def stdWeeklyPropWatched(self, udata):
        """
        @description: Computes the standard deviation of the proportions of videos watched over the weeks. In other words the number of videos watched (only counting the ones assigned)  over the total number of videos assigned.
        @requirement: VideoID, Year (YYYY format), Date (datetime object)
        """
        return weekly_prop_watched(udata).std()

    def avgWeeklyPropReplayed(self, udata):
        """
        @description: Computes the average proportion of videos replayed per week. That is, for each week (nb of videos replayed / nb of videos assigned).
        @requirement: VideoID, Year (YYYY format), Date (datetime object)
        """
        return weekly_prop_replayed(udata).mean()

    def stdWeeklyPropReplayed(self, udata):
        """
        @description: Computes the standard deviation of the proportion of videos replayed over the weeks.
        @requirement: VideoID, Year (YYYY format), Date (datetime object)
        """
        return weekly_prop_replayed(udata).std()

    def avgWeeklyPropInterrupted(self, udata):
        """
        @description: Computes the average proportion of videos interrupted per week. A video is considered interrupted when:
                      * a break is too long
                      * a break (not in the last minute) is followed by an event in another video (the user left the video)
                      * an event occurs in another video before the end of the current video
        @requirement: VideoID, Year (YYYY format), Date (datetime object), EventType, TimeStamp, Duration,
        """
        return weekly_prop_interrupted(udata).mean()

    def stdWeeklyPropInterrupted(self, udata):
        """
        @description: Computes the standard deviation of the proportions of videos interrupted over the weeks.
        @requirement: VideoID, Year (YYYY format), Date (datetime object), EventType, TimeStamp, Duration,
        """
        return weekly_prop_interrupted(udata).std()

    def freqAllActions(self, udata):
        """
        @description: Computes the frequency of actions performed per hour spent watching videos
        @requirement: VideoID, Date (datetime object), Duration
        """
        udata = udata.copy()
        udata.loc[:, 'Day'] = udata.loc[:, 'Date'].dt.date  # Create column with the date but not the time
        udata.drop_duplicates(subset=['VideoID', 'Day'], inplace=True)  # Only keep on event per video per day
        watching_time = udata.Duration.sum() / 3600  # hours
        return total_actions(udata) / watching_time if watching_time != 0 else 0

    def freqPlay(self, udata):
        """
        @description: Computes the ratio of Play events over the total number of actions
        @requirement: VideoID, Date (datetime object), EventType
        """
        return count_actions(udata, 'Video.Play') / total_actions(udata)

    def freqPause(self, udata):
        """
        @description: Computes the ratio of Pause events over the total number of actions
        @requirement: VideoID, Date (datetime object), EventType
        """
        return count_actions(udata, 'Video.Pause') / total_actions(udata)

    def freqSeekBackward(self, udata):
        """
        @description: Computes the ratio of Seek Backward events over the total number of actions
        @requirement: VideoID, Date (datetime object), EventType, OldTime,  NewTime
        """
        return count_actions(udata, 'Video.SeekBackward') / total_actions(udata)

    def freqSeekForward(self, udata):
        """
        @description: Computes the ratio of Seek Forward events over the total number of actions
        @requirement: VideoID, Date (datetime object), EventType, OldTime,  NewTime
        """
        return count_actions(udata, 'Video.SeekForward') / total_actions(udata)

    def freqSpeedChange(self, udata):
        """
        @description: Computes the ratio of SpeedChange events over the total number of actions
        @requirement: VideoID, Date (datetime object), EventType
        """
        return count_actions(udata, 'Video.SpeedChange') / total_actions(udata)

    def freqStop(self, udata):
        """
        @description: Computes the ratio of Stop events over the total number of actions
        @requirement: VideoID, Date (datetime object), EventType
        """
        return count_actions(udata, 'Video.Stop') / total_actions(udata)

    def avgPauseDuration(self, udata):
        """
        @description: Computes the average time interval between each pause event and the next play event. Only pause durations smaller than ~8 min are taken into account.
        @requirement: EventType, TimeStamp
        """
        return pause_duration(udata).mean()

    def stdPauseDuration(self, udata):
        """
        @description: Computes the standard deviation of the time intervals between each pause event and the next play event. Only pause durations smaller than ~8 min are taken into account.
        @requirement: EventType, TimeStamp
        """
        return pause_duration(udata).std()

    def avgSeekLength(self, udata):
        """
        @description: Computes the average seek length. In other words, how much time is skipped forward/backward.
        @requirement: EventType, OldTime, NewTime
        """
        return seek_length(udata).mean()

    def stdSeekLength(self, udata):
        """
        @description: Computes the standard deviation of seek lengths.
        @requirement: EventType, OldTime, NewTime
        """
        return seek_length(udata).std()

    def avgTimeSpeedingUp(self, udata):
        """
        @description: Computes the average time spent at a speed higher than 1 per video.
        @requirement: VideoID, Timestamp, EventType, CurrentTime, Duration
        """
        return compute_time_speeding_up(udata).mean()

    def stdTimeSpeedingUp(self, udata):
        """
        @description: Computes the standard deviation of the time spent at a speed higher than 1 per video.
        @requirement: VideoID, Timestamp, EventType, CurrentTime, Duration
        """
        return compute_time_speeding_up(udata).std()