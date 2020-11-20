from extractors.extractor import Extractor
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
        video_events = udata[udata.EventType.str.contains('Video')]
        problem_events = udata[udata.EventType.str.contains('Problem')]
        features = [self.peakDayHour(udata),
                self.peakWeekDay(udata),
                self.weeklySimilarity1(udata),
                self.weeklySimilarity2(udata),
                self.weeklySimilarity3(udata),
                self.freqDayHour(udata),
                self.freqWeekDay(udata),
                self.freqWeekHour(udata),
                self.nbQuiz(problem_events),
                self.propQuiz(problem_events),
                self.intervalVideoQuiz(video_events, problem_events),
                self.semesterRepartitionQuiz(problem_events)]
        if len(features) != self.getNbFeatures():
            raise Exception(f"getNbFeatures is not up-to-date: {len(features)} != {self.getNbFeatures()}")
        return list(np.nan_to_num(features)) #Set nan to 0