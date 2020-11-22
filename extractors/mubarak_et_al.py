#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractors.extractor import Extractor
import numpy as np

'''
Mubarak, A. A., Cao, H., & Ahmed, S. A. (2020). Predictive learning analytics using deep learning model in MOOCs’ courses videos.
In Education and Information Technologies, 1-22.
'''

class MubarakEtAl(Extractor):

    def __init__(self):
        super().__init__('mubarak_et_al')

    def getNbFeatures(self):
        """Returns the number of features"""
        return 13

    def getUserFeatures(self, udata):

        features = [
            self.fracComp(udata),
            self.fracSpent(udata),
            self.fracPL(udata),
            self.numPL(udata),
            self.numPa(udata),
            self.fracPa(udata),
            self.fracFrwd(udata),
            self.fracBkwd(udata),
            self.numSBW(udata),
            self.numFFW(udata),
            self.avChr(udata),
            self.numLD(udata),
            self.numCompt(udata),
        ]

        if len(features) != self.getNbFeatures():
            raise Exception("getNbFeatures is not up-to-date: {len(features)} != {self.getNbFeatures()}")

        return list(np.nan_to_num(features))

    def fracComp(self, udata):
        """
        @description: The percentage of the video which the learner watched, not counting repeated segment intervals. The completed fraction is a tentative measure of how closely learners are aligned with a video. So, values have to be like [0,1].
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def fracSpent(self, udata):
        """
        @description: The amount of (real) time the learner spent watching the video (i.e. when playing or pausing), compared to the video’s actual duration. FracSpent often reflects the amount of time that it took for the learner to grasp the given content. It can take more than 1 value. And is thus a reflection of the consistency and complexity of the delivery of the material. The high value fracSpent may mean that the l.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def fracPL(self, udata):
        """
        @description: The cumulative amount of play time that the learner was watching, divided by a video’s real duration. Playback periods of under 5 s have been deleted. The fracPL measures repetitive fragments that are watched as opposed to fracComp, which can thus take value greater than 1.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def numPL(self, udata):
        """
        @description: The number of times that video was played by the learner. This function may indicate the learner’s expended efforts to internalize the material further. An abnormally high-value NumPL can therefore suggest that the video quality is unclear, or that the quality is incredibly difficult.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def numPa(self, udata):
        """
        @description: The amount of times the learner has tapped on the pause case in the video. Similar to NumPL, a high value of this attribute suggests the extra effort has been made by the learner to grasp the video material or to perform an off-task conduct.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def fracPa(self, udata):
        """
        @description: The real-time fraction the learner spent pausing on the video, divided on its total playback time. It is possible for fracPa to get up value >1. This feature could take larger values, so that it indicates that the learner exhibited extra effort to understand the material
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def fracFrwd(self, udata):
        """
        @description: The sum of the video the learner had to skip forward while the video was running or pausing; divided on its duration, with repetition. It needs values from 0 to 1.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def fracBkwd(self, udata):
        """
        @description: The sum of time the learner had to skip back when the time was playing or pausing, divided by its duration, with repetition. It takes values from 0 to 1 This feature can take values greater than 1 and may mean that the learner has expended extra time internalizing the video passages. It also indicates that the subject is more interested in or confided in these passages.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def numSBW(self, udata):
        """
        @description: The sum of points in the video the learner was skipping backward. A higher value of this feature indicates the video is interesting or difficult in content.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def numFFW(self, udata):
        """
        @description: The number of times in the video the learner has hopped forward. In this event a high value indicates that the learner is aware that the content of this video clip is not important
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def avChr(self, udata):
        """
        @description: Time-average change of playback Rate selected in the play state by the learner. The Stanford MOOC player allows the default speed rates of between 0.50x and 2.0x. Analysis of average change rate may show that learners viewed the video content quickly or slowly.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def numLD(self, udata):
        """
        @description: The learner has loaded the video the number of times. A value in this feature indicates the learner can watch the video without having to associate it with the content of the course
        @requirement: VideoID, Date (datetime object), EventType
        """
        return

    def numCompt(self, udata):
        """
        @description: The sum of times a learner viewed a video in its entirety. This feature must have a value less than or equal to the number of video playbacks. It also indicates a learner is interested in viewing the whole video.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return