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
from extractors.boroujeni_et_al import BoroujeniEtAl

class Features(unittest.TestCase):
    print("Fetching Video Events...")
    video_df = getVideoEvents()
    
    print("Fetching Problem Events...")
    problem_df = getProblemFirstEvents()


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
        self.assertAlmostEqual(WS1(Lw, T), 0.575, 2)

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
        self.assertAlmostEqual(WS3(Lw, T), 0.18, 2)


    def test_FDH(self):
        sid, T, Lw = getStudentTimeStamps(self.video_df, 44790)
        self.assertAlmostEqual(FDH(Lw, T), 38.86, 2)

        sid, T, Lw = getStudentTimeStamps(self.video_df, 11609)
        self.assertAlmostEqual(FDH(Lw, T), 63.17, 2)


    def test_FWH(self):
        sid, T, Lw = getStudentTimeStamps(self.video_df, 44790)
        self.assertAlmostEqual(FWH(Lw, T), 45.33, 2)

        sid, T, Lw = getStudentTimeStamps(self.video_df, 11609)
        self.assertAlmostEqual(FWH(Lw, T), 44.47, 2)


    def test_FWD(self):
        sid, T, Lw = getStudentTimeStamps(self.video_df, 44790)
        self.assertAlmostEqual(FWD(Lw, T), 13.09, 2)
        
        sid, T, Lw = getStudentTimeStamps(self.video_df, 11609)
        self.assertAlmostEqual(FWD(Lw, T), 8.73, 2)


    def test_NQZ(self):
        self.assertEqual(NQZ(self.problem_df, 8438), 193)
        self.assertEqual(NQZ(self.problem_df, 46953), 213)
        self.assertEqual(NQZ(self.problem_df, 46968), 55)

    def test_PQZ(self):
        self.assertAlmostEqual(PQZ(self.problem_df, 46485), 0.16, 2)
        self.assertAlmostEqual(PQZ(self.problem_df, 12275), 0.88, 2)
        self.assertAlmostEqual(PQZ(self.problem_df, 18422), 0.89, 2)

    def test_IVQ(self):
        PATH_DATED = '../data/lin_alg_moodle/'
        PATH_DATED_VIDEOS = PATH_DATED + 'videos.csv'
        PATH_DATED_PROBLEMS = PATH_DATED + 'problems.csv'

        dated_videos_df = pd.read_csv(PATH_DATED_VIDEOS, index_col=0)
        dated_problems_df = pd.read_csv(PATH_DATED_PROBLEMS, index_col=0)
        ivq = IVQ(46497, self.video_df, self.problem_df, dated_videos_df, dated_problems_df)
        self.assertAlmostEqual(ivq, 1.77, 2)

    def test_SRQ(self):
        srq = SRQ(12275, self.problem_df)
        self.assertAlmostEqual(srq, 33, 0)


if __name__ == '__main__':
    unittest.main()