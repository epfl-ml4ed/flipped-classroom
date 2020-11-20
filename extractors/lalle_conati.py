from extractors.extractor import Extractor

'''
Lallé, S., & Conati, C. (2020, July). A Data-Driven Student Model to Provide Adaptive Support During Video Watching Across MOOCs.
In International Conference on Artificial Intelligence in Education (pp. 282-295). Springer, Cham.
'''

class LalleConati(Extractor):
    def __init__(self):
        super().__init__('lalle_conati')

    def getUserFeatures(self, udata):
        return [self.totalViews(udata),
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