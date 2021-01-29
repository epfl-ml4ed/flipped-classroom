#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import pandas as pd
from tqdm import tqdm

from helper.database.db_connector import MySQLConnector


def downloadEvents(config):
    db = MySQLConnector()
    for platform in config['platforms']:
        print('loading', config['table'].lower(), 'from', platform)
        courses = pd.DataFrame.from_records(db.execute("""SELECT DISTINCT DataPackageID FROM {}.{}""".format(platform, config['table'])), columns=['DataPackageID'])['DataPackageID'].tolist()
        print('>', platform, 'includes', len(courses), 'courses with', config['table'].lower())
        for course in tqdm(set(courses) & config(['courses']) if config['courses'] is not None else set(courses)):
            if not os.path.exists(os.path.join(config['table'].lower(), config['table'].lower() + '_' + platform + '_' + course + '.csv')):
                courseEventsDf = pd.DataFrame.from_records(db.execute("""SELECT {} FROM {}.{} WHERE DataPackageID = \'{}\'""".format(', '.join(config['columns']), platform, config['table'], course)), columns=config['columns'])
                courseEventsDf['Platform'] = platform
                if 'coursera' in platform:
                    courseEventsDf['AccountUserID'] = courseEventsDf['SessionUserID']
                    del courseEventsDf['SessionUserID']
                courseEventsDf.to_csv(os.path.join(config['table'].lower(), config['table'].lower() + '_' + platform + '_' + course + '.csv'), index=False)
        print('ended parsing from', platform)
    db.close()

def main():

    config = {
        'table': 'Request_2021Jan26_Problem_Events',
        'courses': None,
        'columns': ['DataPackageID', 'ProblemID', 'AccountUserHash', 'EventType', 'TimeStamp', 'ProblemType', 'Grade', 'SubmissionNumber'],
        'platforms': ['project_himanshu']
    }

    downloadEvents(config)

    config = {
        'table': 'Request_2021Jan26_Video_Events',
        'courses': None,
        'columns': ['DataPackageID', 'VideoID', 'AccountUserHash', 'EventType', 'TimeStamp', 'SeekType', 'OldTime', 'CurrentTime', 'NewTime', 'OldSpeed', 'NewSpeed'],
        'platforms': ['project_himanshu']
    }

    downloadEvents(config)

if __name__ == '__main__':
    main()