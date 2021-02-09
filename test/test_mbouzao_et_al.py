import unittest
import pandas as pd
from helper.hcourse import init_courses
from extractor.feature.feature import Feature
from extractor.feature.attendance_rate import AttendanceRate
from extractor.feature.utilization_rate import UtilizationRate
from extractor.feature.watching_index import WatchingIndex

class TestHeEtAl(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        assert Feature.TIME_MIN == 1.0 and Feature.TIME_MAX == 3600
        cls.course = init_courses({'types': ['toy-course'], 'course_ids': ['toy_course-20210202_000840'],
                                   'load': True, 'label': True})[0]
        cls.feature_settings = {
            'model': 'extractor.set.mbouzao_et_al.MbouzaoEtAl',
            'courses': 'toy-course/toy_course-20210202_000840',
            'workdir': '../data/result/test/feature',
            'course': cls.course
        }

        cls.events = pd.read_csv('../data/course/toy-course/toy-platform/video_event/toy_course-20210202_000840.csv')
        cls.events['date'] = pd.to_datetime(cls.events.date)

    # tests for attendance rate and utilization rate in test_he_et_al.py

    def test_WatchIndex(self):
        settings = self.feature_settings.copy()
        settings.update({'timeframe': 'eq_week', 'week': 0})
        data = self.events.query('(user_id == 1) & (week == 0)')
        attendance_rate = AttendanceRate(data, settings).compute()
        utilization_rate = UtilizationRate(data, settings).compute()
        effective_value = WatchingIndex(data, settings).compute()
        self.assertEqual(effective_value, attendance_rate * utilization_rate)

if __name__ == '__main__':
    unittest.main()
