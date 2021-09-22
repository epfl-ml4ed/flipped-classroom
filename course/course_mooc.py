#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import logging
import os

from course.course import Course

class CourseMOOC(Course):

    def __init__(self, id, type, platform):
        super().__init__(id, type, platform)

    def label(self, labels=None, week_thr=None):
        assert self.clickstream_grade is not None

        if labels is None or 'label-grade' in labels:
            self.clickstream_grade['label-grade'] = self.clickstream_grade['grade']
            logging.info('assigned grades')

        if labels is None or 'label-pass-fail' in labels:
            self.clickstream_grade['label-pass-fail'] = self.clickstream_grade['pass-fail'].apply(lambda x: 0 if x == 'Passed' else 1)
            logging.info('assigned {} pass and {} fail'.format(self.clickstream_grade['label-pass-fail'].to_list().count(0), self.clickstream_grade['label-pass-fail'].to_list().count(1), 'drop'))

        if labels is None or 'label-dropout' in labels:
            week_thr = week_thr if week_thr is not None else self.weeks
            max_week = self.get_clickstream_video().groupby(by='user_id')['week'].max()
            self.clickstream_grade['label-dropout'] = np.array([(1 if row['user_id'] in max_week and max_week[row['user_id']] < week_thr - 1 else 0) for index, row in self.clickstream_grade.iterrows()])
            logging.info('assigned {} no-drop and {} drop'.format(self.clickstream_grade['label-dropout'].to_list().count(0), self.clickstream_grade['label-dropout'].to_list().count(1), 'drop'))

        if labels is None or 'label-stopout' in labels:
            max_week = self.get_clickstream_video().groupby(by='user_id')['week'].max()
            self.clickstream_grade['label-stopout'] = np.array([(max_week[row['user_id']] if row['user_id'] in max_week else 0) for index, row in self.clickstream_grade.iterrows()])
            logging.info('assigned stopout')