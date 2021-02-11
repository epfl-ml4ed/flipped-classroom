import unittest
import logging
import numpy as np
import pandas as pd
from helper.hcourse import init_courses
from extractor.feature.feature import Feature
from extractor.feature.fraction_spent import FractionSpent
from extractor.feature.frequency_event import FrequencyEvent


class TestMubarakEtAl(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        assert Feature.TIME_MIN == 1.0 and Feature.TIME_MAX == 3600
        cls.course = init_courses({'types': ['toy-course'], 'course_ids': ['toy_course-20210202_000840'],
                                   'load': True, 'label': True})[0]
        cls.feature_settings = {
            'model': 'extractor.set.mubarak_et_al.MubarakEtAl',
            'courses': 'toy-course/toy_course-20210202_000840',
            'workdir': '../data/result/test/feature',
            'course': cls.course
        }
        cls.schedule = cls.course.get_video_schedule()
        cls.events = pd.read_csv('../data/course/toy-course/toy-platform/video_event/toy_course-20210202_000840.csv')
        cls.events['date'] = pd.to_datetime(cls.events.date)
        cls.events['weekday'] = cls.events.date.dt.weekday

    # SpeedPlayback is tested in test_lemay_doleck.py
    # As well as other modes of FractionSpent and FrequencyEvent

    def test_FractionSpent(self):
        # Only test mode = time
        settings = self.feature_settings.copy()
        settings.update({'timeframe': 'full', 'mode': 'time', 'type': 'Video.Seek'})
        data = self.events.query('(user_id == 1) & (video_id == 2)')
        phases = ['forward', 'backward']
        expected_values = [
            np.array([269 - 241, 283 - 269, 303 - 283, 264 - 233.32700, 276 - 265.03900, 295 - 276.43400, 314 - 295.44600]) / 817,
            np.array([300.201 - 278, 278 - 269, 269 - 260, 260 - 253, 253 - 241, 312.65300 - 230, 568.21800 - 564, 567.68900 - 551]) / 817]
        for i, phase in enumerate(phases):
            settings['phase'] = phase
            self.assertAlmostEqual(np.sum(expected_values[i]), FractionSpent(data, settings).compute())

    def test_FrequencyEvent(self):
        settings = self.feature_settings.copy()
        settings.update({'timeframe': 'full', 'mode': 'relative'})
        data = self.events.query('user_id == 72').copy()

        for event_type in ['Video.Play', 'Video.Pause']:
            settings['type'] = event_type
            expected_value = len(data.query('event_type == @event_type')) / data.video_id.nunique()
            self.assertEqual(expected_value, FrequencyEvent(data, settings).compute())


if __name__ == '__main__':
    unittest.main()