import unittest
import logging
import numpy as np
import pandas as pd
from helper.hcourse import init_courses
from extractor.feature.feature import Feature
from extractor.feature.time import Time
from extractor.feature.frequency_event import FrequencyEvent
from extractor.feature.fraction_spent import FractionSpent
from extractor.feature.speed_playback import SpeedPlayback
from extractor.feature.count_unique_element import CountUniqueElement

class TestLemayDoleck(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        assert Feature.TIME_MIN == 1.0 and Feature.TIME_MAX == 3600
        cls.course = init_courses({'types': ['toy-course'], 'course_ids': ['toy_course-20210202_000840'],
                                   'load': True, 'label': True})[0]
        cls.feature_settings = {
            'model': 'extractor.set.lemay_doleck.LemayDoleck',
            'courses': 'toy-course/toy_course-20210202_000840',
            'workdir': '../data/result/test/feature',
            'course': cls.course
        }
        cls.schedule = cls.course.get_video_schedule()
        cls.events = pd.read_csv('../data/course/toy-course/toy-platform/video_event/toy_course-20210202_000840.csv')
        cls.events['date'] = pd.to_datetime(cls.events.date)
        cls.events['weekday'] = cls.events.date.dt.weekday

    def test_FractionSpent(self):
        settings = self.feature_settings.copy()
        settings.update({'timeframe': 'lq_week', 'week': 7, 'ffunc': np.sum})
        data = self.events.query('(user_id == 72) & (video_id == 63)').drop_duplicates('timestamp')

        for event_type in ['Video.Play', 'Video.Pause']:
            settings['type'] = event_type
            effective_value = FractionSpent(data, settings).compute()
            expected_value = Time(data, settings).compute()
            expected_value /= self.schedule.query('id == 63').duration.values[0]
            self.assertEqual(expected_value, effective_value)

        settings['type'] = 'video'
        expected_value = Time(data, settings).compute()
        expected_value /= self.schedule.query('id == 63').duration.values[0]
        settings.update({'type': 'Video.Play', 'mode': 'played'})
        self.assertEqual(expected_value, FractionSpent(data, settings).compute())

        settings.update({'type': 'Video.Play', 'mode': 'completed'})
        self.assertEqual((676 - 34) / 676, FractionSpent(data, settings).compute())

    def test_FrequencyEvent(self):
        settings = self.feature_settings.copy()
        settings.update({'timeframe': 'lq_week', 'week': 7, 'mode': 'total'})
        data = self.events.query('user_id == 72').drop_duplicates('timestamp')
        data['event_type'] = np.where(data.old_time > data.new_time, 'Video.SeekBackward', data.event_type)
        data['event_type'] = np.where(data.old_time < data.new_time, 'Video.SeekForward', data.event_type)

        for event_type in ['Video.Pause', 'Video.SeekBackward', 'Video.SeekForward']:
            settings['type'] = event_type
            effective_value = FrequencyEvent(data, settings).compute()
            expected_value = len(data.query('(week <= 7) & (event_type == @event_type)'))
            self.assertEqual(expected_value, effective_value)

    def test_SpeedPlayback(self):
        settings = self.feature_settings.copy()
        settings.update({'timeframe': 'full'})
        data = self.events.query('user_id == 56').copy()
        data['new_speed'].fillna(method='ffill', inplace=True)
        data['new_speed'].fillna(data.old_speed.fillna(method='bfill'), inplace=True)
        speeds = data.new_speed.values
        for ffunc in [np.mean, np.std]:
            settings['ffunc'] = ffunc
            self.assertEqual(ffunc(speeds), SpeedPlayback(data, settings).compute())

    def test_CountUniqueElement(self):
        settings = self.feature_settings.copy()
        settings.update({'timeframe': 'eq_week', 'week': 4})
        problems = pd.read_csv('../data/course/toy-course/toy-platform/problem_event/toy_course-20210202_000840.csv')
        data = pd.concat([self.events, problems]).query('user_id == 72')
        data['date'] = pd.to_datetime(data.date)

        for event_type in ['video', 'problem']:
            settings['type'] = event_type
            effective_value = CountUniqueElement(data, settings).compute()
            expected_value = data.query('(week == 4) ')[event_type + '_id'].nunique()
            self.assertEqual(expected_value, effective_value)


if __name__ == '__main__':
    unittest.main()
