import unittest
import os
import sys
import numpy as np
from datetime import datetime, timedelta
# import parent folder
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from helpers.feature_extraction import *
from helpers.db_query import *

class Features(unittest.TestCase):
    print("Fetching Video Events...")
    video_events = getVideoEvents(mode='all')

    def test_total_views(self):
        user_events1 = self.video_events[self.video_events.AccountUserID == '18422']
        user_events2 = self.video_events[self.video_events.AccountUserID == '100402']
        self.assertEqual(total_views(user_events1), 131)
        self.assertEqual(total_views(user_events2), 99)

    def test_avg_weekly_prop_watched(self):
        user_events1 = self.video_events[self.video_events.AccountUserID == '18422']
        user_events2 = self.video_events[self.video_events.AccountUserID == '100402']
        self.assertAlmostEqual(avg_weekly_prop_watched(user_events1),  0.6972405372405373,2)
        self.assertAlmostEqual(avg_weekly_prop_watched(user_events2),   0.8102564102564103,2)

    def test_std_weekly_prop_watched(self):
        user_events1 = self.video_events[self.video_events.AccountUserID == '18422']
        user_events2 = self.video_events[self.video_events.AccountUserID == '100402']
        self.assertAlmostEqual(std_weekly_prop_watched(user_events1),  0.37290747472253855,2)
        self.assertAlmostEqual(std_weekly_prop_watched(user_events2),   0.3562349688859857,2)

    def test_avg_weekly_prop_replayed(self):
        user_events1 = self.video_events[self.video_events.AccountUserID == '18422']
        user_events2 = self.video_events[self.video_events.AccountUserID == '100402']
        self.assertAlmostEqual(avg_weekly_prop_replayed(user_events1),  0.08987484737484737,2)
        self.assertAlmostEqual(avg_weekly_prop_replayed(user_events2),   0.01952662721893491,2)

    def test_std_weekly_prop_replayed(self):
        user_events1 = self.video_events[self.video_events.AccountUserID == '18422']
        user_events2 = self.video_events[self.video_events.AccountUserID == '100402']
        self.assertAlmostEqual(std_weekly_prop_replayed(user_events1),  0.09161292670783444,2)
        self.assertAlmostEqual(std_weekly_prop_replayed(user_events2),   0.04699580982642552,2)

    def test_avg_weekly_prop_interrupted(self):
        user_events1 = self.video_events[self.video_events.AccountUserID == '18422']
        user_events2 = self.video_events[self.video_events.AccountUserID == '100402']
        self.assertAlmostEqual(avg_weekly_prop_interrupted(user_events1),  0.1345848595848596,2)
        self.assertAlmostEqual(avg_weekly_prop_interrupted(user_events2),   0.044378698224852076,2)

    def test_std_weekly_prop_interrupted(self):
        user_events1 = self.video_events[self.video_events.AccountUserID == '18422']
        user_events2 = self.video_events[self.video_events.AccountUserID == '100402']
        self.assertAlmostEqual(std_weekly_prop_interrupted(user_events1),  0.13206607167995404,2)
        self.assertAlmostEqual(std_weekly_prop_interrupted(user_events2),   0.10752962548260876,2)

    def test_total_actions(self):
        user_events1 = self.video_events[self.video_events.AccountUserID == '18422']
        user_events2 = self.video_events[self.video_events.AccountUserID == '100402']
        self.assertEqual(total_actions(user_events1),  4065)
        self.assertEqual(total_actions(user_events2),   1405)

    def test_frequency_all_actions(self):
        user_events1 = self.video_events[self.video_events.AccountUserID == '18422']
        user_events2 = self.video_events[self.video_events.AccountUserID == '100402']
        self.assertAlmostEqual(frequency_all_actions(user_events1),  4.6784199298727405,2)
        self.assertAlmostEqual(frequency_all_actions(user_events2),  4.6219963780847095,2)

    def test_freq_play(self):
        user_events1 = self.video_events[self.video_events.AccountUserID == '18422']
        user_events2 = self.video_events[self.video_events.AccountUserID == '100402']
        self.assertAlmostEqual(freq_play(user_events1),  0.43739237392373925,2)
        self.assertAlmostEqual(freq_play(user_events2),   0.3302491103202847,2)

    def test_freq_pause(self):
        user_events1 = self.video_events[self.video_events.AccountUserID == '18422']
        user_events2 = self.video_events[self.video_events.AccountUserID == '100402']
        self.assertAlmostEqual(freq_pause(user_events1),  0.4150061500615006,2)
        self.assertAlmostEqual(freq_pause(user_events2),   0.07473309608540925,2)

    def test_freq_seek_backward(self):
        user_events1 = self.video_events[self.video_events.AccountUserID == '18422']
        user_events2 = self.video_events[self.video_events.AccountUserID == '100402']
        self.assertAlmostEqual(freq_seek_backward(user_events1),  0.012054120541205412,2)
        self.assertAlmostEqual(freq_seek_backward(user_events2),   0.22633451957295372,2)

    def test_freq_seek_forward(self):
        user_events1 = self.video_events[self.video_events.AccountUserID == '18422']
        user_events2 = self.video_events[self.video_events.AccountUserID == '100402']
        self.assertAlmostEqual(freq_seek_forward(user_events1),  0.009102091020910209,2)
        self.assertAlmostEqual(freq_seek_forward(user_events2),   0.022775800711743774,2)

    def test_freq_speed_change(self):
        user_events1 = self.video_events[self.video_events.AccountUserID == '18422']
        user_events2 = self.video_events[self.video_events.AccountUserID == '100402']
        self.assertAlmostEqual(freq_speed_change(user_events1),  0.005412054120541206,2)
        self.assertAlmostEqual(freq_speed_change(user_events2),   0.12740213523131672,2)

    def test_freq_stop(self):
        user_events1 = self.video_events[self.video_events.AccountUserID == '18422']
        user_events2 = self.video_events[self.video_events.AccountUserID == '100402']
        self.assertAlmostEqual(freq_stop(user_events1),  0.02779827798277983,2)
        self.assertAlmostEqual(freq_stop(user_events2),   0.04412811387900356,2)

    def test_avg_pause_duration(self):
        user_events1 = self.video_events[self.video_events.AccountUserID == '18422']
        user_events2 = self.video_events[self.video_events.AccountUserID == '100402']
        self.assertAlmostEqual(avg_pause_duration(user_events1),  47.85451197053407,2)
        self.assertAlmostEqual(avg_pause_duration(user_events2),   82.98936170212765,2)

    def test_std_pause_duration(self):
        user_events1 = self.video_events[self.video_events.AccountUserID == '18422']
        user_events2 = self.video_events[self.video_events.AccountUserID == '100402']
        self.assertAlmostEqual(std_pause_duration(user_events1),  80.07571048872693,2)
        self.assertAlmostEqual(std_pause_duration(user_events2),  89.54785891221961,2)

    def test_avg_seek_length(self):
        user_events1 = self.video_events[self.video_events.AccountUserID == '18422']
        user_events2 = self.video_events[self.video_events.AccountUserID == '100402']
        self.assertAlmostEqual(avg_seek_length(user_events1),  173.77237999999997,2)
        self.assertAlmostEqual(avg_seek_length(user_events2),  44.90929303977273,2)

    def test_std_seek_length(self):
        user_events1 = self.video_events[self.video_events.AccountUserID == '18422']
        user_events2 = self.video_events[self.video_events.AccountUserID == '100402']
        self.assertAlmostEqual(std_seek_length(user_events1),  230.00152378749962,2)
        self.assertAlmostEqual(std_seek_length(user_events2),  49.21778945794014,2)

    def test_avg_time_speeding_up(self):
        user_events1 = self.video_events[self.video_events.AccountUserID == '18422']
        user_events2 = self.video_events[self.video_events.AccountUserID == '100402']
        self.assertAlmostEqual(avg_time_speeding_up(user_events1),  500.79916326315794,2)
        self.assertAlmostEqual(avg_time_speeding_up(user_events2),  334.5920517934782,2)

    def test_std_time_speeding_up(self):
        user_events1 = self.video_events[self.video_events.AccountUserID == '18422']
        user_events2 = self.video_events[self.video_events.AccountUserID == '100402']
        self.assertAlmostEqual(std_time_speeding_up(user_events1),  424.287219725475,2)
        self.assertAlmostEqual(std_time_speeding_up(user_events2),  229.1368767069532,2)



    # total_views
    # avg_weekly_prop_watched
    # std_weekly_prop_watched
    # avg_weekly_prop_replayed
    # std_weekly_prop_replayed
    # avg_weekly_prop_interrupted
    # std_weekly_prop_interrupted
    # total_actions
    # frequency_all_actions
    # freq_play
    # freq_pause
    # freq_seek_backward
    # freq_seek_forward
    # freq_speed_change
    # freq_stop
    # avg_pause_duration
    # std_pause_duration
    # avg_seek_length
    # std_seek_length
    # avg_time_speeding_up
    # std_time_speeding_up
if __name__ == '__main__':
    unittest.main()