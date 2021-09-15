#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.extractor import Extractor

from extractor.feature.competency_strength import CompetencyStrength
from extractor.feature.competency_alignment import CompetencyAlignment
from extractor.feature.competency_coverage import CompetencyCoverage
from extractor.feature.competency_anticipation import CompetencyAnticipation
from extractor.feature.content_alignment import ContentAlignment
from extractor.feature.content_coverage import ContentCoverage
from extractor.feature.content_anticipation import ContentAnticipation
from extractor.feature.student_speed import StudentSpeed
from extractor.feature.student_shape import StudentShape
from extractor.feature.student_activeness import StudentActiveness
from extractor.feature.student_thoughtfulness import StudentThoughtfulness
from extractor.feature.student_weekly_activeness import StudentWeeklyActiveness

'''
Marras, M., Vignoud, J. T. T., & Käser, T.
Can Feature Predictive Power Generalize? Benchmarking Early Predictors of
Student Success across Flipped and Online Courses.
'''

class MarrasEtAl(Extractor):
    def __init__(self, name='base'):
        super().__init__(name)
        self.name = 'marras_et_al'

    def extract_features(self, data, settings):
        self.features = [CompetencyStrength(data, settings),
                         CompetencyAlignment(data, settings),
                         CompetencyCoverage(data, settings),
                         CompetencyAnticipation(data, settings),
                         ContentAlignment(data, settings),
                         ContentCoverage(data, settings),
                         ContentAnticipation(data, settings),
                         StudentSpeed(data, settings),
                         StudentShape(data, settings),
                         StudentActiveness(data, settings),
                         StudentThoughtfulness(data, settings),
                         StudentWeeklyActiveness(data, settings)]

        features = [f.compute() for f in self.features]
        assert len(features) == self.__len__()
        return features
