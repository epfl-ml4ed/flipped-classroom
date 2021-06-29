import unittest
import numpy as np
import pandas as pd
from helper.hcourse import init_courses
from extractor.feature.feature import Feature
from extractor.feature.number_sessions import NumberSessions
from extractor.feature.time_sessions import TimeSessions
from extractor.feature.time_between_sessions import TimeBetweenSessions
from extractor.feature.ratio_clicks_weekend_day import RatioClicksWeekendDay
from extractor.feature.time import Time


class TestChenCui(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        assert Feature.TIME_MIN == 1.0 and Feature.TIME_MAX == 3600
        cls.course = init_courses({'types': ['toy-course'], 'course_ids': ['toy_course-20210202_000840'],
                                   'load': True, 'label': True})[0]
        cls.feature_settings = {
            'model': 'extractor.set.chen_cui.ChenCui',
            'courses': 'toy-course/toy_course-20210202_000840',
            'workdir': '../data/result/test/feature',
            'course': cls.course
        }

        cls.events = pd.read_csv('../data/course/toy-course/toy-platform/video_event/toy_course-20210202_000840.csv')
        cls.events['date'] = pd.to_datetime(cls.events.date)
        cls.events['weekday'] = cls.events.date.dt.weekday

    # Tests for Total clicks implemented in test_lalle_conati.py

    def test_NumberSessions(self):
        settings = self.feature_settings.copy()
        settings.update({'timeframe': 'lq_week', 'week': 1})
        data = self.events.query('user_id == 1')
        self.assertEqual(NumberSessions(data, settings).compute(), 6)

    def test_TimeSessions(self):
        settings = self.feature_settings.copy()
        settings.update({'timeframe': 'lq_week', 'week': 0})
        data = self.events.query('user_id == 1')
        session_lengths = [1571081009 - 1571077210, 1571258102 - 1571254966]

        for ffunc in [np.sum, np.mean, np.std]:
            settings['ffunc'] = ffunc
            self.assertEqual(TimeSessions(data, settings).compute(), ffunc(session_lengths))

    def test_TimeBetweenSessions(self):
        settings = self.feature_settings.copy()
        settings.update({'timeframe': 'lq_week', 'week': 1})
        data = self.events.query('user_id == 1')
        time_intersessions = np.abs([1571081009 - 1571254966, 1571258102 - 1571670882, 1571671673 - 1571674514,
                                     1571677513 - 1571681011, 1571687683 - 1571860619])
        settings['ffunc'] = np.std
        self.assertEqual(TimeBetweenSessions(data, settings).compute(), np.std(time_intersessions))

    def test_RatioClicksWeekendDay(self):
        settings = self.feature_settings.copy()
        settings.update({'timeframe': 'lq_week', 'week': 7})
        data = self.events.query('user_id == 72 & week < 8')
        expected_value = sum(data['weekday'].isin(range(5))) / sum(data['weekday'].isin([5, 6]))
        self.assertEqual(RatioClicksWeekendDay(data, settings).compute(), expected_value)

    def test_Time(self):
        settings = self.feature_settings.copy()
        settings.update({'timeframe': 'eq_week', 'week': 0, 'type': 'problem', 'ffunc': np.sum})

        problems = pd.read_csv('../data/course/toy-course/toy-platform/problem_event/toy_course-20210202_000840.csv')
        data = pd.concat([self.events, problems])
        data['date'] = pd.to_datetime(data.date)
        data['weekday'] = data.date.dt.weekday
        data = data.query('(user_id == 1) & (timestamp <= 1571078320)')

        problem_expected_value = np.sum(np.abs([1571076871 - 1571077210, 1571078237 - 1571078320]))
        self.assertEqual(Time(data, settings).compute(), problem_expected_value)


if __name__ == '__main__':
    unittest.main()
