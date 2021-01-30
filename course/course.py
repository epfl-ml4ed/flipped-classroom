#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import logging
import os

from helper.htime import init_clickstream, init_schedule

class Course():

    def __init__(self, id, type, platform):
        self.course_id = id
        self.type = type
        self.platform = platform

    def load(self, filepath=os.path.join(os.path.abspath(os.path.dirname(__file__)), '../data/course')):
        metadata_path = os.path.join(filepath, self.type, 'metadata.csv')
        if os.path.exists(metadata_path):
            entire_metadata = pd.read_csv(metadata_path)
            course_metadata = entire_metadata[entire_metadata['course_id'] == self.course_id].to_dict(orient='records')
            if len(course_metadata) > 0:
                for key, value in course_metadata[0].items():
                    setattr(self, key, value)
                logging.info('loaded metadata for {}'.format(self.course_id))
            else:
                logging.warning('metadata missing for {}'.format(self.course_id))

        grade_path = os.path.join(filepath, self.type, self.platform, 'grade', self.course_id + '.csv')
        if os.path.exists(grade_path):
            self.clickstream_grade = pd.read_csv(grade_path)
            logging.info('loaded grades for {}'.format(self.course_id))
        else:
            logging.warning('grades missing for {}'.format(self.course_id))

        video_path = os.path.join(filepath, self.type, self.platform, 'video_event', self.course_id + '.csv')
        if os.path.exists(video_path):
            data = init_clickstream(pd.read_csv(video_path), self.type, self.start_date, self.end_date)
            self.clickstream_video = data[data['user_id'].isin(self.clickstream_grade['user_id'].unique())]
            logging.info('loaded video events for {}'.format(self.course_id))
        else:
            logging.warning('video events missing for {}'.format(self.course_id))

        problem_path = os.path.join(filepath, self.type, self.platform, 'problem_event', self.course_id + '.csv')
        if os.path.exists(problem_path):
            data = init_clickstream(pd.read_csv(problem_path), self.type, self.start_date, self.end_date)
            self.clickstream_problem = data[data['user_id'].isin(self.clickstream_grade['user_id'].unique())]
            logging.info('loaded problem events for {}'.format(self.course_id))
        else:
            logging.warning('problem events missing for {}'.format(self.course_id))

        schedule_path = os.path.join(filepath, self.type, self.platform, 'schedule', self.course_id + '.csv')
        if os.path.exists(schedule_path):
            schedule = pd.read_csv(schedule_path)
            if len(schedule) > 0:
                self.schedule = init_schedule(schedule, self.type, self.start_date, self.end_date)
                logging.info('loaded schedule for {}'.format(self.course_id))
            else:
                logging.warning('schedule missing for {}'.format(self.course_id))

    def label(self, labels=None, week_thr=None):
        assert self.clickstream_grade is not None and self.grade_thr is not None and self.grade_max is not None

        if labels is None or 'pass-fail' in labels:
            self.clickstream_grade['pass-fail'] = self.clickstream_grade['grade'].apply(lambda x: 1 if x < self.grade_thr * self.grade_max else 0)
            logging.info('assigned {} pass and {} fail'.format(self.clickstream_grade['pass-fail'].to_list().count(0), self.clickstream_grade['pass-fail'].to_list().count(1), 'drop'))

        if labels is None or 'dropout' in labels:
            week_thr = week_thr if week_thr is not None else self.weeks
            max_week = self.get_clickstream_video().groupby(by='user_id')['week'].max()
            self.clickstream_grade['dropout'] = np.array([(1 if row['user_id'] in max_week and max_week[row['user_id']] < week_thr - 1 else 0) for index, row in self.clickstream_grade.iterrows()])
            logging.info('assigned {} no-drop and {} drop'.format(self.clickstream_grade['dropout'].to_list().count(0), self.clickstream_grade['dropout'].to_list().count(1), 'drop'))

        if labels is None or 'stopout' in labels:
            max_week = self.get_clickstream_video().groupby(by='user_id')['week'].max()
            self.clickstream_grade['stopout'] = np.array([max_week[row['user_id']] for index, row in self.clickstream_grade.iterrows()])
            logging.info('assigned stopout')

    def get_weeks(self):
        assert self.weeks is not None
        return self.weeks

    def get_clickstream(self):
        assert self.clickstream_video is not None or self.clickstream_problem is not None
        return self.clickstream_video.copy() if self.clickstream_problem is None else self.clickstream_video.append(self.clickstream_problem).copy()

    def get_clickstream_problem(self):
        assert self.clickstream_problem is not None
        return self.clickstream_problem.copy()

    def get_clickstream_video(self):
        assert self.clickstream_video is not None
        return self.clickstream_video.copy()

    def get_clickstream_grade(self):
        assert self.clickstream_grade is not None
        return self.clickstream_grade.copy()

    def get_schedule(self):
        assert self.schedule is not None
        return self.schedule.copy()

    def is_complete(self):
        return self.metadata is not None and self.schedule is not None and self.clickstream_grade is not None and self.clickstream_video is not None

    def has_schedule(self):
        return self.schedule is not None

    def __str__(self):
        return 'ID: {} Type: {} Title: {} Students: {}'.format(self.course_id, self.type, self.title, self.__len__())

    def __add__(self, x):
        self.clickstream_grade = self.clickstream_grade.append(x.clickstream_grade)
        self.clickstream_video = self.clickstream_grade.append(x.clickstream_video)
        self.clickstream_problem = self.clickstream_grade.append(x.clickstream_problem)
        self.schedule = self.clickstream_grade.append(x.schedule)
        return self

    def __len__(self):
        assert self.clickstream_video is not None
        return len(self.clickstream_grade['user_id'].unique())