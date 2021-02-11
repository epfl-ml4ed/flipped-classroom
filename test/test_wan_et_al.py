import unittest
import logging
import numpy as np
import pandas as pd
from helper.hcourse import init_courses
from extractor.feature.feature import Feature
from extractor.feature.number_submissions import NumberSubmissions
from extractor.feature.time import Time
from extractor.feature.time_sessions import TimeSessions
from extractor.feature.obs_duration_problem import ObsDurationProblem
from extractor.feature.time_solve_problem import TimeSolveProblem


class TestWanEtAl(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        assert Feature.TIME_MIN == 1.0 and Feature.TIME_MAX == 3600
        cls.course = init_courses({'types': ['toy-course'], 'course_ids': ['toy_course-20210202_000840'],
                                   'load': True, 'label': True})[0]
        cls.feature_settings = {
            'model': 'extractor.set.wan_et_al.WanEtAl',
            'courses': 'toy-course/toy_course-20210202_000840',
            'workdir': '../data/result/test/feature',
            'course': cls.course
        }
        cls.schedule = cls.course.get_video_schedule()
        cls.events = pd.read_csv('../data/course/toy-course/toy-platform/video_event/toy_course-20210202_000840.csv')
        cls.events['date'] = pd.to_datetime(cls.events.date)
        cls.events['weekday'] = cls.events.date.dt.weekday

        cls.problems = pd.read_csv(
            '../data/course/toy-course/toy-platform/problem_event/toy_course-20210202_000840.csv')
        cls.problems['date'] = pd.to_datetime(cls.problems.date)

    def test_TimeSessions(self):
        #  Only mode = length case covered here as TimeSessions already tested in test_chen_cui.py
        settings = self.feature_settings.copy()
        settings.update({'timeframe': 'eq_week', 'week': 3, 'mode': 'length'})
        data = self.events.query('user_id == 1')
        expected_value = 4
        effective_value = TimeSessions(data, settings).compute()
        self.assertEqual(expected_value, effective_value)

    def test_NumberSubmissions(self):
        settings = self.feature_settings.copy()
        settings.update({'timeframe': 'eq_week', 'week': 3})
        data = self.problems.query('(user_id == 1) & (week == 3)').copy()
        self.assertEqual(10, NumberSubmissions(data, settings).compute())

        modes = ['distinct', 'distinct_correct', 'avg', 'avg_time', 'perc_correct', 'correct']
        filtered_data = data[(data.event_type == 'Problem.Check') & (~data.grade.isna())].copy()
        filtered_data['time_diff'] = filtered_data.timestamp.diff()
        filtered_data['prev_id'] = filtered_data.problem_id.shift(1)
        time_intervals = filtered_data.query('(prev_id == problem_id) &'
                                             ' (time_diff >= @Feature.TIME_MIN) & '
                                             '(time_diff <= @Feature.TIME_MAX)').time_diff.values

        expected_values = [8,
                           filtered_data[filtered_data.grade == 100].problem_id.nunique(),
                           filtered_data.groupby('problem_id').size().mean(),
                           time_intervals.mean(),
                           7 / 10,
                           8 / 6]
        for mode, expected_value in zip(modes, expected_values):
            settings['mode'] = mode
            self.assertEqual(expected_value, NumberSubmissions(data, settings).compute())

    def test_ObsDurationProblem(self):
        settings = self.feature_settings.copy()
        settings.update({'timeframe': 'full'})
        data = self.problems.query('user_id == 72').copy()
        expected_value = TimeSessions(data, {**settings, **{'ffunc': np.sum}}).compute() / \
                         NumberSubmissions(data, {**settings, **{'mode': 'distinct_correct'}}).compute()
        self.assertEqual(expected_value, ObsDurationProblem(data, settings).compute())

        data['prev_id'] = data.problem_id.shift(1)
        data['time_diff'] = data.timestamp.diff()
        time_intervals = data.query('(prev_id == problem_id) &'
                                    ' (time_diff >= @Feature.TIME_MIN) & '
                                    '(time_diff <= @Feature.TIME_MAX)').time_diff.values
        for ffunc in [np.var, np.max]:
            settings['ffunc'] = ffunc
            self.assertEqual(ffunc(time_intervals), ObsDurationProblem(data, settings).compute())

    def test_TimeSolveProblem(self):
        settings = self.feature_settings.copy()
        settings.update({'timeframe': 'full'})
        data = self.problems.query('user_id == 72').sort_values('timestamp').copy()
        data['prev_id'] = data.problem_id.shift(1)
        data = pd.concat([data.drop_duplicates('problem_id', keep='first'), data.drop_duplicates('problem_id', keep='last')])
        data = data.query('prev_id == problem_id')
        data = data.sort_values('timestamp')
        data['time_diff'] = data.sort_values('timestamp').timestamp.diff()
        expected_value = data.query('(time_diff >= @Feature.TIME_MIN) & (time_diff <= @Feature.TIME_MAX)').time_diff.mean()

        self.assertEqual(expected_value, TimeSolveProblem(data, settings).compute())


if __name__ == '__main__':
    unittest.main()
