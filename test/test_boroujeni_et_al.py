import unittest
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from extractor.extractor_loader import ExtractorLoader
from helper.hcourse import init_courses
from extractor.feature.reg_peak_time import RegPeakTime
from extractor.feature.reg_weekly_sim import RegWeeklySim
from extractor.feature.reg_periodicity import RegPeriodicity
from extractor.feature.delay_lecture import DelayLecture

# logging.getLogger().setLevel(logging.INFO)

def load_set(settings):
    # Load feature set
    extractor = ExtractorLoader()
    extractor.load(settings)

    # Arrange data
    feature_labels = extractor.get_features_values()[0][settings['target']].values
    feature_values = extractor.get_features_values()[1]

    y = feature_labels if settings['target_type'] == 'regression' else feature_labels.astype(int)
    X = feature_values

    return X, y


class TestBoroujeni(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        eq_settings = {
            'feature_set': 'eq_week-boroujeni_et_al-toy_course_20210202_000840-20210202_160217',
            'target': 'label-pass-fail',
            'target_type': 'classification',
            'classes': 1,
            'workdir': '../data/result/test'
        }
        lq_settings = {
            'feature_set': 'lq_week-boroujeni_et_al-toy_course_20210202_000840-20210202_163428',
            'target': 'label-pass-fail',
            'target_type': 'classification',
            'classes': 1,
            'workdir': '../data/result/test'
        }
        cls.X_eq, cls.y_eq = load_set(eq_settings)
        cls.X_lq, cls.y_lq = load_set(lq_settings)

        cls.course = init_courses({'types': ['toy-course'], 'course_ids': ['toy_course-20210202_000840'],
                                   'load': True, 'label': True})[0]
        cls.feature_settings = {
            'model': 'extractor.set.boroujeni_et_al.BoroujeniEtAl',
            'courses': 'toy-course/toy_course-20210202_000840',
            'workdir': '../data/result/test/feature',
            'course': cls.course
        }

        initial_dt = datetime(2021, 2, 3, 15)

        def create_uniform_hours():
            # one timestamp each hour
            data = [[initial_dt + timedelta(hours=i), 0, 0] for i in range(24)]
            df = pd.DataFrame(data, columns=['date', 'week', 'user_id'])
            return df

        def create_skewed_hours():
            # The same hour over one week
            data = [[initial_dt + timedelta(days=i), 0, 0] for i in range(7)]
            df = pd.DataFrame(data, columns=['date', 'week', 'user_id'])
            return df

        cls.uniform_data = create_uniform_hours()
        cls.skewed_data = create_skewed_hours()

    def test_dimensions(self):
        self.assertEqual(self.X_eq.shape, (9, 12, 3))
        self.assertEqual(self.y_eq.shape, (9,))

        self.assertEqual(self.X_lq.shape, (9, 12, 9))
        self.assertEqual(self.y_lq.shape, (9,))

    def test_RegPeakTime(self):
        dayhour_settings = self.feature_settings.copy()
        dayhour_settings.update({'mode': 'dayhour', 'week': 0, 'timeframe': 'eq_week'})

        # PDH on uniform and skewed data
        uniform_value = RegPeakTime(self.uniform_data, dayhour_settings).compute()
        self.assertAlmostEqual(uniform_value, 0, 5)

        skewed_value = RegPeakTime(self.skewed_data, dayhour_settings).compute()
        self.assertAlmostEqual(skewed_value, np.log(24) * 7, 5)

        # PWD on skewed data
        weekday_settings = self.feature_settings.copy()
        weekday_settings.update({'mode': 'weekday', 'week': 1, 'timeframe': 'lq_week'})
        weekday_value = RegPeakTime(self.skewed_data, weekday_settings).compute()
        self.assertAlmostEqual(weekday_value, 0, 5)

    def test_RegPeriodicity(self):
        m1_settings = self.feature_settings.copy()
        m1_settings.update({'mode': 'm1', 'week': 0, 'timeframe': 'eq_week'})

        # Mode 1 (FDH) on uniform data
        df = self.uniform_data.copy()
        df['weekday'] = df.date.dt.weekday
        df['event_type'] = 'dummy'
        uniform_value = RegPeriodicity(df, m1_settings).compute()
        self.assertAlmostEqual(uniform_value, 0, 5)

        # Mode 1 (FDH) on single timestamp
        unique_df = pd.DataFrame([[datetime(2021, 2, 3, 12), 0, 0]], columns=['date', 'week', 'user_id'])
        unique_df['weekday'] = unique_df.date.dt.weekday
        unique_df['event_type'] = 'dummy'
        unique_value = RegPeriodicity(unique_df, m1_settings).compute()
        self.assertAlmostEqual(unique_value, 1, 5)

        # Mode 2 (FWH) on 2 timestamps at same hour one week apart
        m2_settings = self.feature_settings.copy()
        m2_settings.update({'mode': 'm2', 'week': 1, 'timeframe': 'lq_week'})
        data = [[datetime(2021, 2, 3, 12) + timedelta(weeks=i), i, 0] for i in range(2)]
        m2_df = pd.DataFrame(data, columns=['date', 'week', 'user_id'])
        m2_df['weekday'] = unique_df.date.dt.weekday
        m2_df['event_type'] = 'dummy'
        m2_value = RegPeriodicity(m2_df, m2_settings).compute()
        self.assertAlmostEqual(m2_value, 2, 5)

        # Mode 3 (FWD) on 2 timestamps at same hour one week apart
        m3_settings = self.feature_settings.copy()
        m3_settings.update({'mode': 'm3', 'week': 1, 'timeframe': 'lq_week'})
        m3_value = RegPeriodicity(m2_df, m3_settings).compute()
        self.assertAlmostEqual(m3_value, 2, 5)

    def test_DelayLecture(self):
        events = pd.read_csv('../data/course/toy-course/toy-platform/video_event/toy_course-20210202_000840.csv')
        events['date'] = pd.to_datetime(events.date)
        data = events.query('user_id == 1')

        settings = {**self.feature_settings, **{'timeframe': 'full', 'week': 14}}
        effective_value = DelayLecture(data, settings).compute()

        schedule = pd.read_csv('../data/course/toy-course/toy-platform/schedule/toy_course-20210202_000840.csv')
        schedule['date'] = pd.to_datetime(schedule.date)
        schedule = schedule.query('type == "video"')
        data = data.sort_values(by='date').drop_duplicates(['video_id'])
        delays = data.merge(schedule, left_on='video_id', right_on='id')
        expected_value = (delays.date_x - delays.date_y).mean().total_seconds()

        self.assertAlmostEqual(effective_value, expected_value, 5)

    def test_RegWeeklySim(self):
        # Mode 1 (WS1)
        m1_settings = self.feature_settings.copy()
        m1_settings.update({'timeframe': 'lq_week', 'week': 1, 'mode': 'm1'})

        m1_data = [[datetime(2021, 1, 2, 12), 0, 0],  # in common between 2 weeks
                   [datetime(2021, 1, 3, 1), 0, 0],
                   [datetime(2021, 1, 9, 12), 1, 0],  # in common between 2 weeks
                   [datetime(2021, 1, 11, 1), 1, 0]]
        m1_data = pd.DataFrame(m1_data, columns=['date', 'week', 'user_id'])
        m1_data['weekday'] = m1_data.date.dt.weekday
        m1_value = RegWeeklySim(m1_data, m1_settings).compute()
        self.assertEqual(m1_value, 0.5)

        # Mode 2 (WS2)
        m2_settings = self.feature_settings.copy()
        m2_settings.update({'timeframe': 'lq_week', 'week': 4, 'mode': 'm2'})
        # One hour active every day for 5 weeks
        m2_data = [[datetime(2021, 1, 2) + timedelta(weeks=week, days=day), week, 0] for week in range(5) for day in range(7)]
        m2_data = pd.DataFrame(m2_data, columns=['date', 'week', 'user_id'])
        m2_data['weekday'] = m2_data.date.dt.weekday
        m2_value = RegWeeklySim(m2_data, m2_settings).compute()
        self.assertEqual(m2_value, 1)

        m2_data.drop_duplicates('week', inplace=True) # Keep one day active per week (the same day)
        m2_value = RegWeeklySim(m2_data, m2_settings).compute()
        self.assertEqual(m2_value, 1)

        # Mode 3 (WS3)
        m3_settings = m2_settings.copy()
        m3_settings['mode'] = 'm3'
        m3_data = [[datetime(2021, 1, 2, 0), 0, 0],
                   [datetime(2021, 1, 2, 1), 0, 0],
                   [datetime(2021, 1, 9, 0), 1, 0]]
        m3_data = pd.DataFrame(m3_data, columns=['date', 'week', 'user_id'])
        m3_data['weekday'] = m3_data.date.dt.weekday
        m3_value = RegWeeklySim(m3_data, m3_settings).compute()
        self.assertAlmostEqual(m3_value, 8/9, 5)





if __name__ == '__main__':
    unittest.main()
