#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from course.course import Course
from course.course_mooc import CourseMOOC

import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def init_courses(settings, filepath=os.path.join(os.path.abspath(os.path.dirname(__file__)), '../data/course')):
    courses = []
    for type in os.listdir(filepath):
        if 'types' in settings and type not in settings['types']:
            continue
        for platform in os.listdir(os.path.join(filepath, type)):
            if not os.path.isdir(os.path.join(filepath, type, platform)) or 'platform' in settings is not None and platform not in settings['platform']:
                continue
            for course_file in os.listdir(os.path.join(filepath, type, platform, 'video_event')):
                course_id = os.path.splitext(course_file)[0]
                if 'course_ids' in settings is not None and course_id not in settings['course_ids']:
                    continue
                if type == 'mooc':
                    course = CourseMOOC(course_id, type, platform)
                else:
                    course = Course(course_id, type, platform)
                if settings['load']:
                    course.load()
                if settings['label']:
                    course.label()
                courses.append(course)
    logging.info('loaded {} courses'.format(len(courses)))
    return courses

def find_course_by_id(course_id, course_lst):
    for c in course_lst:
        if c.course_id == course_id:
            return c
    return None

def main():
    courses = init_courses({'types': ['flipped-classroom'], 'course_ids': ['EPFL-AlgebreLineaire-2018', 'EPFL-AlgebreLineaire-2019'], 'load': True, 'label': True})
    c = courses[0] + courses[1]
    print(len(c))
    print('First', find_course_by_id('EPFL-AlgebreLineaire-2018', courses))
    print('Second', find_course_by_id('EPFL-AlgebreLineaire-2020', courses))

if __name__ == "__main__":
    main()