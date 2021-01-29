#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from course.course import Course

class CourseISA(Course):

    def __init__(self, id, title, topic, type, platform, start_date=None, end_date=None, grade_thr=0.80):
        super().__init__(id, title, topic, type, platform, start_date, end_date, grade_thr)

    def add_fail_flag(self):
        self.clickstream_grade['pass-fail'] = self.clickstream_grade['grade'].apply(lambda x: 1 if x < 4.0 else 0)
