import unittest
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from helper.hcourse import init_courses
from extractor.feature.attendance_rate import AttendanceRate
from extractor.feature.utilization_rate import UtilizationRate
from extractor.feature.watching_ratio import WatchingRatio


# logging.getLogger().setLevel(logging.INFO)

class TestHeEtAl(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.course = init_courses({'types': ['toy-course'], 'course_ids': ['toy_course-20210202_000840'],
                                   'load': True, 'label': True})[0]
        cls.feature_settings = {
            'model': 'extractor.set.he_et_al.HeEtAl',
            'courses': 'toy-course/toy_course-20210202_000840',
            'workdir': '../data/result/test/feature',
            'course': cls.course
        }

        cls.events = pd.read_csv('../data/course/toy-course/toy-platform/video_event/toy_course-20210202_000840.csv')
        cls.events['date'] = pd.to_datetime(cls.events.date)
        cls.schedule = pd.read_csv('../data/course/toy-course/toy-platform/schedule/toy_course-20210202_000840.csv')
        cls.schedule['date'] = pd.to_datetime(cls.schedule.date)
        cls.schedule = cls.schedule.query('type == "video"')

    def test_AttendanceRate(self):

        # leq with 3 weeks
        settings = self.feature_settings.copy()
        settings.update({'timeframe': 'lq_week', 'week': 3})
        data = self.events.query('(user_id == 1) & (week <= 3)')
        effective_value = AttendanceRate(data, settings).compute()
        watched_videos = set(data.video_id.unique())
        taught_videos = set(self.schedule.query('week <= 3').id)
        expected_value = len(watched_videos & taught_videos) / len(taught_videos)
        self.assertAlmostEqual(effective_value, expected_value, 5)

        settings['timeframe'] = 'eq_week'
        data = self.events.query('(user_id == 1) & (week == 3)')
        effective_value = AttendanceRate(data, settings).compute()
        watched_videos = set(data.video_id.unique())
        taught_videos = set(self.schedule.query('week == 3').id)
        expected_value = len(watched_videos & taught_videos) / len(taught_videos)
        self.assertAlmostEqual(effective_value, expected_value, 5)

        # over the whole semester
        settings['timeframe'] = 'full'
        data = self.events.query('user_id == 1')
        effective_value = AttendanceRate(data, settings).compute()
        watched_videos = set(data.video_id.unique())
        taught_videos = set(self.schedule.id)
        expected_value = len(watched_videos & taught_videos) / len(taught_videos)
        self.assertAlmostEqual(effective_value, expected_value, 5)

    def test_UtilizationRate(self):
        settings = self.feature_settings.copy()
        settings.update({'timeframe': 'eq_week', 'week': 0})
        data = self.events.query('(user_id == 1) & (week == 0)').head(10)
        effective_value = UtilizationRate(data, settings).compute()

        sum_video_length = self.schedule.query('week == 0').duration.sum()
        sum_time_intervals = 208 + 195 # values found by inspection of the dataframe
        self.assertAlmostEqual(effective_value, sum_time_intervals / sum_video_length, 5)

    def test_WatchRatio(self):
        settings = self.feature_settings.copy()
        settings.update({'timeframe': 'eq_week', 'week': 0})
        data = self.events.query('(user_id == 1) & (week == 0)')
        attendance_rate = AttendanceRate(data, settings).compute()
        utilization_rate = UtilizationRate(data, settings).compute()
        effective_value = WatchingRatio(data, settings).compute()
        self.assertEqual(effective_value, attendance_rate/utilization_rate)

        settings.update({'timeframe': 'full'})
        data = self.events.query('user_id == 1')
        attendance_rate = AttendanceRate(data, settings).compute()
        utilization_rate = UtilizationRate(data, settings).compute()
        effective_value = WatchingRatio(data, settings).compute()
        self.assertAlmostEqual(effective_value, attendance_rate/utilization_rate, 2)

if __name__ == '__main__':
    unittest.main()
