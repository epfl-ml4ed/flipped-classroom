from extractors.extractor import Extractor
import numpy as np
'''
Lallé, S., & Conati, C. (2020, July). A Data-Driven Student Model to Provide Adaptive Support During Video Watching Across MOOCs.
In International Conference on Artificial Intelligence in Education (pp. 282-295). Springer, Cham.
'''

class LalleConati(Extractor):
    def __init__(self):
        super().__init__('lalle_conati')

    def getNbFeatures(self):
        """Returns the number of features"""
        return 21
    
    def getUserFeatures(self, udata):
        features = [self.totalViews(udata),
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
                self.avgTimeSpeeding_up(udata),
                self.stdTimeSpeedingUp(udata)]

        if len(features) != self.getNbFeatures():
            raise Exception(f"getNbFeatures is not up-to-date: {len(features)} != {self.getNbFeatures()}")
        return list(np.nan_to_num(features)) #Set nan to 0