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
from helpers.data_process import getStudentTimeStamps
from helpers.db_query import *

class Features(unittest.TestCase):
    print("Fetching Video Events...")
    video_df = getVideoEvents()

    def test_PDH(self):
        # PDH bounded in [0, log(24) * L_d]

        # Particular sid
        sid, T, Lw = getStudentTimeStamps(self.video_df, 11609)
        self.assertAlmostEqual(PDH(Lw, T), 23.37, 2) #2 decimals accuracy

        #Lower bound
        T = []
        Lw = 1
        self.assertEqual(PDH(Lw, T), 0)

        #Upper bound log(24) * L_d
        #All the activity at the same hour
        Lw = 10
        T = np.arange(Lw) * 3600 * 24
        self.assertAlmostEqual(PDH(Lw, T), np.log2(24) * Lw, 2) 

    # PWD bounded in [0, log(7) * L_w]
    def test_PWD(self):  
        # Particular sid
        sid, T, Lw = getStudentTimeStamps(self.video_df, 9749)
        self.assertAlmostEqual(PWD(Lw, T), 7.51, 2)
        #Lower bound
        T = []
        Lw = 1
        self.assertEqual(PWD(Lw, T), 0)

        #Upper bound log(7) * L_w
        #All the activity on the same day
        Lw = 10
        T = np.arange(Lw) * 3600 * 24 * 7
        self.assertAlmostEqual(PWD(Lw, T), np.log2(7) * Lw, 2)

    def test_WS1(self):
        sid, T, Lw = getStudentTimeStamps(self.video_df, 10118)
        self.assertAlmostEqual(WS1(Lw, T), 0.57, 1)

        sid, T, Lw = getStudentTimeStamps(self.video_df, 46587)
        self.assertAlmostEqual(WS1(Lw, T), 0.12, 2)

    def test_WS2(self):
        sid, T, Lw = getStudentTimeStamps(self.video_df, 10118)
        self.assertAlmostEqual(WS2(Lw, T), 0.49, 2)

        sid, T, Lw = getStudentTimeStamps(self.video_df, 46587)
        self.assertAlmostEqual(WS2(Lw, T), 0.19, 2)

    def test_WS3(self):
        sid, T, Lw = getStudentTimeStamps(self.video_df, 10118)
        self.assertAlmostEqual(WS3(Lw, T), 0.50, 2)

        sid, T, Lw = getStudentTimeStamps(self.video_df, 46587)
        self.assertAlmostEqual(WS3(Lw, T), 0.19, 0.18)



if __name__ == '__main__':
    unittest.main()