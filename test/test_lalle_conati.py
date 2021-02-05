import unittest
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from helper.hcourse import init_courses
from extractor.feature.total_clicks import TotalClicks
from extractor.feature.seek_length import SeekLength
from extractor.feature.frequency_event import FrequencyEvent
from extractor.feature.weekly_prop import WeeklyProp
from extractor.feature.pause_duration import PauseDuration
from extractor.feature.time_speeding_up import TimeSpeedingUp


# logging.getLogger().setLevel(logging.INFO)

class TestLalleConati(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.course = init_courses({'types': ['toy-course'], 'course_ids': ['toy_course-20210202_000840'],
                                   'load': True, 'label': True})[0]
        cls.feature_settings = {
            'model': 'extractor.set.lalle_conati.LalleConati',
            'courses': 'toy-course/toy_course-20210202_000840',
            'workdir': '../data/result/test/feature',
            'course': cls.course
        }

        cls.events = pd.read_csv('../data/course/toy-course/toy-platform/video_event/toy_course-20210202_000840.csv')
        cls.events['date'] = pd.to_datetime(cls.events.date)
        cls.events['weekday'] = cls.events.date.dt.weekday

    def test_TotalClicks(self):
        settings = self.feature_settings.copy()

        settings.update({'timeframe': 'full'})
        data = self.events.query('user_id == 56')
        effective_value = TotalClicks(data, settings).compute()
        self.assertEqual(effective_value, len(data))

        settings['mode'] = 'weekend'
        effective_value = TotalClicks(data, settings).compute()
        self.assertEqual(effective_value, len(data[data.weekday.isin([5, 6])]))

        settings['mode'] = 'weekday'
        effective_value = TotalClicks(data, settings).compute()
        self.assertEqual(effective_value, len(data[~data.weekday.isin([5, 6])]))

        del settings['mode']
        settings['type'] = 'Video'
        effective_value = TotalClicks(data, settings).compute()
        self.assertEqual(effective_value, len(data[data.event_type.str.contains("Video")]))

        settings['type'] = 'Problem'
        effective_value = TotalClicks(data, settings).compute()
        self.assertEqual(effective_value, len(data[data.event_type.str.contains("Problem")]))

    def test_WeeklyProp(self):
        settings = self.feature_settings.copy()
        settings.update({'timeframe': 'eq_week', 'week': 0, 'ffunc': np.mean})
        data = self.events.query('user_id == 72')
        videos_to_watch = 7  # 7 videos to watch in week 0

        # Proportion of videos watched
        settings['type'] = 'watched'
        effective_value = WeeklyProp(data, settings).compute()
        videos_watched_in_time = 5  # user watched 5 of the videos scheduled for the first week
        self.assertEqual(effective_value, videos_watched_in_time / videos_to_watch)

        # Proportion of videos replayed (if same videos watched on different days)
        settings['type'] = 'replayed'
        effective_value = WeeklyProp(data, settings).compute()
        videos_replayed = 1  # video_id 55
        self.assertEqual(effective_value, videos_replayed / videos_to_watch)

        # Proportion of videos interrupted
        settings['type'] = 'interrupted'
        settings['week'] = 3
        effective_value = WeeklyProp(data, settings).compute()
        videos_interrupted = 1  # video_id 16 due to long break
        videos_to_watch_week3 = 13
        self.assertEqual(effective_value, videos_interrupted / videos_to_watch_week3)

    def test_FrequencyEvent(self):
        settings = self.feature_settings.copy()

        settings.update({'timeframe': 'full'})
        data = self.events.query('user_id == 56')
        settings['type'] = 'Video'
        # TODO: make this assertion true
        # self.assertNotEqual(FrequencyEvent(data, settings).compute(), 1.0)

        for event in data.event_type.unique():
            settings['type'] = event
            effective_value = FrequencyEvent(data, settings).compute()
            expected_value = len(data[data.event_type.str.contains(event)]) / len(data)
            self.assertEqual(effective_value, expected_value)

    def test_SeekLength(self):
        pass

    def test_PauseDuration(self):
        pass

    def test_TimeSpeedingUp(self):
        pass

if __name__ == '__main__':
    unittest.main()
