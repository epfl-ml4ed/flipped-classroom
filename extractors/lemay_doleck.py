from extractors.extractor import Extractor

'''
Lemay, D. J., & Doleck, T. (2020). Grade prediction of weekly assignments in MOOCS: mining video-viewing behavior.
Education and Information Technologies, 25(2), 1333-1342.
'''

class LemayDoleck(Extractor):
    def __init__(self):
        super().__init__('lemay_doleck')

    def getUserFeatures(self, udata):
        return [
            self.totalEvents(udata, 'Video.Play', unique=True),
            self.totalEvents(udata, 'Video.Pause'),
            self.totalEvents(udata, 'Video.Seek'),
            self.totalEvents(udata, 'Video.SpeedChange'),
        ]