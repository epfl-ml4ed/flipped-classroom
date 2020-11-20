from extractors.extractor import Extractor
import numpy as np
'''
Lemay, D. J., & Doleck, T. (2020). Grade prediction of weekly assignments in MOOCS: mining video-viewing behavior.
Education and Information Technologies, 25(2), 1333-1342.
'''

class LemayDoleck(Extractor):
    def __init__(self):
        super().__init__('lemay_doleck')

    def getNbFeatures(self):
        """Returns the number of features"""
        return 4

    def getUserFeatures(self, udata):
        features =  [
            self.totalEvents(udata, 'Video.Play', unique=True),
            self.totalEvents(udata, 'Video.Pause'),
            self.totalEvents(udata, 'Video.Seek'),
            self.totalEvents(udata, 'Video.SpeedChange'),
        ]

        if len(features) != self.getNbFeatures():
            raise Exception(f"getNbFeatures is not up-to-date: {len(features)} != {self.getNbFeatures()}")
        return list(np.nan_to_num(features)) #Set nan to 0