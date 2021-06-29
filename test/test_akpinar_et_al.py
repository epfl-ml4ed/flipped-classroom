import unittest
import logging
import numpy as np
import pandas as pd
from helper.hcourse import init_courses
from extractor.feature.feature import Feature
from extractor.feature.count_ngrams import CountNGrams


class TestAkpinarEtAl(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        assert Feature.TIME_MIN == 1.0 and Feature.TIME_MAX == 3600
        cls.course = init_courses({'types': ['toy-course'], 'course_ids': ['toy_course-20210202_000840'],
                                   'load': True, 'label': True})[0]
        cls.feature_settings = {
            'model': 'extractor.set.akpinar_et_al.AkpinarEtAl',
            'courses': 'toy-course/toy_course-20210202_000840',
            'workdir': '../data/result/test/feature',
            'course': cls.course
        }
        cls.schedule = cls.course.get_video_schedule()
        cls.events = pd.read_csv('../data/course/toy-course/toy-platform/video_event/toy_course-20210202_000840.csv')
        cls.events['date'] = pd.to_datetime(cls.events.date)
        cls.events['weekday'] = cls.events.date.dt.weekday

    # TotalClicks is tested in test_lalle_conati.py
    # NumberSessions and Time are tested in test_chen_cui

    def test_CountNGrams(self):
        settings = self.feature_settings.copy()
        settings['timeframe'] = 'full'
        data = self.events.query('(user_id == 72) & (timestamp >= 1571035031) & (timestamp <= 1571035768)')
        expected_value = [2, 2, 1, 2, 2, 1, 1, 1, 1, 1]
        effective_value = CountNGrams(data, settings).compute()
        self.assertEqual(expected_value, effective_value)


if __name__ == '__main__':
    unittest.main()
