#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import logging

from helper.hcourse import init_courses
from helper.hutils import import_class

logging.getLogger().setLevel(logging.INFO)

def main(settings):

    # Check course
    courses_lst = settings['course'].split(',')
    courses_types = list(set([c.split('/')[0] for c in courses_lst]))
    courses_ids = list(set([c.split('/')[1] for c in courses_lst]))
    courses = init_courses({'types': courses_types, 'course_ids': courses_ids, 'load': True, 'label': True})

    assert len(courses) == 1

    time_str = time.strftime('%Y%m%d_%H%M%S')

    # Load course

    course = courses[0]
    logging.info('found course - {}'.format(settings['course']))

    video_event = course.get_clickstream_video()
    problem_event = course.get_clickstream_problem()
    grade = course.get_clickstream_grade()
    schedule = course.get_schedule()

    # Mapping ids
    schedule_problem_event = schedule[schedule['type'] == 'problem']
    schedule_video_event = schedule[schedule['type'] == 'video']

    video_event = video_event[video_event['video_id'].isin(schedule_video_event['id'])]
    problem_event = problem_event[problem_event['problem_id'].isin(schedule_problem_event['id'])]

    user_id_int_map = dict(zip(video_event.user_id, video_event.user_id.astype('category').cat.codes))
    video_id_int_map = dict(zip(schedule_video_event.id, schedule_video_event.id.astype('category').cat.codes))
    problem_id_int_map = dict(zip(schedule_problem_event.id, schedule_problem_event.id.astype('category').cat.codes))

    video_event['user_id'] = video_event['user_id'].apply(lambda x: user_id_int_map[x])
    video_event['video_id'] = video_event['video_id'].apply(lambda x: video_id_int_map[x])
    problem_event['user_id'] = problem_event['user_id'].apply(lambda x: user_id_int_map[x])
    problem_event['problem_id'] = problem_event['problem_id'].apply(lambda x: problem_id_int_map[x])
    grade['user_id'] = grade['user_id'].apply(lambda x: user_id_int_map[x])
    schedule['id'] = schedule['id'].apply(lambda x: problem_id_int_map[x] if x in problem_id_int_map else video_id_int_map[x])

    # Sample users

    users = np.random.choice(grade['user_id'].unique(), settings['no_users'])
    video_event = video_event[video_event['user_id'].isin(users)]
    problem_event = problem_event[problem_event['user_id'].isin(users)]
    grade = grade[grade['user_id'].isin(users)]

    # Save data

    video_path = os.path.join(settings['workdir'], 'toy-course', 'toy-platform', 'video_event')
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    video_event.to_csv(os.path.join(video_path, 'toy_course-' + time_str + '.csv'), index=False)

    problem_path = os.path.join(settings['workdir'], 'toy-course', 'toy-platform', 'problem_event')
    if not os.path.exists(problem_path):
        os.makedirs(problem_path)
    problem_event.to_csv(os.path.join(problem_path, 'toy_course-' + time_str + '.csv'), index=False)

    grade_path = os.path.join(settings['workdir'], 'toy-course', 'toy-platform', 'grade')
    if not os.path.exists(grade_path):
        os.makedirs(grade_path)
    grade.to_csv(os.path.join(grade_path, 'toy_course-' + time_str + '.csv'), index=False)

    schedule_path = os.path.join(settings['workdir'], 'toy-course', 'toy-platform', 'schedule')
    if not os.path.exists(schedule_path):
        os.makedirs(schedule_path)
    schedule.to_csv(os.path.join(schedule_path, 'toy_course-' + time_str + '.csv'), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract feature')

    parser.add_argument('--course', dest='course', default='flipped-classroom/EPFL-AlgebreLineaire-2019', type=str, action='store')
    parser.add_argument('--workdir', dest='workdir', default='../data/course/', type=str, action='store')
    parser.add_argument('--no_users', dest='no_users', default=10, type=int, action='store')

    settings = vars(parser.parse_args())

    main(settings)