#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from extractor.extractor import Extractor


class ChenCui(Extractor):

    def __init__(self, name='base'):
        """
        @description: Returns the identifier associated with this feature set.
        """
        super().__init__('chen_cui')

    def getNbFeatures(self):
        """
        @description: Returns the number of expected features.
        """
        return len(self.getFeatureNames())

    def getFeatureNames(self):
        """
        @description: Returns the names of the feature in the same order as getUserFeatures
        """
        return ["totalClicks", "numberSessions", "totalTimeAllSessions", "avgSessionTime", "stdSessionTime", "totalClicksWeekdays",
                "totalClicksWeekends", "ratioClicksWeekdaysWeekends", "totalClicksOnProblems", "totalTimeOnProblems", "stdTimeOnProblems"]

    def getUserFeatures(self, udata, config):
        """
        @description: Returns the user features computed from the udata
        """

        udata = udata.copy()
        udata = udata.sort_values(by='TimeStamp')
        udata['Weekday'] = udata['TimeStamp'].apply(lambda x: 1 if x.weekday() < 5 else 0)

        features = [
            self.totalClicks(udata, config),
            self.numberSessions(udata, config),
            self.totalTimeAllSessions(udata, config),
            self.avgSessionTime(udata, config),
            self.stdSessionTime(udata, config),
            self.totalClicksWeekdays(udata, config),
            self.totalClicksWeekends(udata, config),
            self.ratioClicksWeekdaysWeekends(udata, config),
            self.totalClicksOnProblems(udata, config),
            self.totalTimeOnProblems(udata, config),
            self.stdTimeOnProblems(udata, config)
        ]

        if len(features) != self.getNbFeatures():
            raise Exception("getNbFeatures is not up-to-date: {len(features)} != {self.getNbFeatures()}")

        return list(np.nan_to_num(features))

    def totalClicks(self, udata, config):
        """
        @description: The number of total clicks.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return len(udata.index)

    def numberSessions(self, udata, config):
        """
        @description: The number of online sessions
        @requirement: VideoID, Date (datetime object), EventType
        """
        return len(getSessions(udata, config))

    def totalTimeAllSessions(self, udata, config):
        """
        @description: The total time for all online sessions.
        @requirement: VideoID, Date (datetime object), EventType
        """
        sessions = getSessions(udata, config)
        return np.sum(sessions['Duration'])

    def avgSessionTime(self, udata, config):
        """
        @description: The mean of online session time.
        @requirement: VideoID, Date (datetime object), EventType
        """
        sessions = getSessions(udata, config)
        return np.mean(sessions['Duration'])

    def stdSessionTime(self, udata, config):
        """
        @description: The standard deviation of online session time.
        @requirement: VideoID, Date (datetime object), EventType
        """
        sessions = getSessions(udata, config)
        return np.std(sessions['Duration'])

    def totalClicksWeekdays(self, udata, config):
        """
        @description: The number of clicks during weekdays.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return len(udata[udata['Weekday'] == 1].index)

    def totalClicksWeekends(self, udata, config):
        """
        @description: The number of clicks during weekends.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return len(udata[udata['Weekday'] == 0].index)

    def ratioClicksWeekdaysWeekends(self, udata, config):
        """
        @description: The ratio of weekend to weekday clicks
        @requirement: VideoID, Date (datetime object), EventType
        """
        return self.totalClicksWeekdays(udata, config) / (self.totalClicksWeekends(udata, config) + sys.float_info.epsilon)

    def totalClicksOnProblems(self, udata, config):
        """
        @description: The number of clicks for module “Quiz”.
        @requirement: VideoID, Date (datetime object), EventType
        """
        return len(udata[udata['EventType'].str.contains('Problem.')].index)

    def totalTimeOnProblems(self, udata, config):
        """
        @description: the total time on module “Quiz”.
        @requirement: VideoID, Date (datetime object), EventType
        """
        udata['PrevEvent'] = udata['EventType'].shift(1)
        udata['PrevProblemID'] = udata['ProblemID'].shift(1)
        udata['TimeDiff'] = udata.TimeStamp.diff()
        udata['TimeDiff'] = udata['TimeDiff'].apply(lambda x : x.total_seconds())
        udata = udata[(udata['PrevEvent'].str.contains('Problem.')) & (udata['ProblemID'] == udata['PrevProblemID'])]
        return np.sum(udata['TimeDiff'])

    def stdTimeOnProblems(self, udata, config):
        """
        @description: The standard deviation of time on module “Quiz”.
        @requirement: VideoID, Date (datetime object), EventType
        """
        udata['PrevEvent'] = udata['EventType'].shift(1)
        udata['PrevProblemID'] = udata['ProblemID'].shift(1)
        udata['TimeDiff'] = udata.TimeStamp.diff()
        udata['TimeDiff'] = udata['TimeDiff'].apply(lambda x : x.total_seconds())
        udata = udata[(udata['PrevEvent'].str.contains('Problem.')) & (udata['ProblemID'] == udata['PrevProblemID'])]
        return np.std(udata['TimeDiff'])

warnings.filterwarnings('ignore')

def getSeconds(row):
    if 'Start' in row and 'End' in row:
        return (row['End'] - row['Start']).total_seconds()
    return 0.0

def string2Datetime(ts):
    return datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')

def datetime2Week(dt):
    return dt.strftime('%W')

def dropDateOutsiders(series, start_date, end_date):
    start_dt = string2Datetime(start_date)
    end_dt = string2Datetime(end_date)
    return series[(series >= start_dt) & (series <= end_dt)].values

def orderWeek(series, start_date):
    start_week = int(datetime2Week(string2Datetime(start_date)))
    nb_week = int(series.max())
    return series.apply(lambda w: str((int(w) - start_week) % nb_week)).values

def processWeek(data, column, start_date):
    data['Week'] = data[column].apply(datetime2Week)
    data['Week'] = orderWeek(data['Week'], start_date).astype(int)
    return data.sort_values(by=column)

def getSessions(config, metadata):
    sessionDict = {}

    courses = config['courses'] if config['courses'] is not None else ['_'.join(x.split('.')[0].split('_')[2:]) for x in os.listdir('video_events')]

    for course in courses:

        videos = pd.read_csv(os.path.join('video_events', 'video_events_' + course + '.csv'))
        videos['ElementID'] = videos['VideoID']
        df = videos[['AccountUserID', 'ElementID', 'TimeStamp', 'EventType']]
        print('loading', len(df), 'video events from', len(videos['AccountUserID'].unique()),'students for', course, '...')

        if os.path.exists(os.path.join('problem_events', 'problem_events_' + course + '.csv')):
            problems = pd.read_csv(os.path.join('problem_events', 'problem_events_' + course + '.csv'))
            problems['ElementID'] = problems['ProblemID']
            df = df.append(problems[['AccountUserID', 'ElementID', 'TimeStamp', 'EventType']])
            print('loading', len(problems), 'problem events from', len(problems['AccountUserID'].unique()),'students for', course, '...')
        else:
            print('no problem events found for', course)

        df['TimeStamp'] = df['TimeStamp'].apply(lambda x: string2Datetime(datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S')))
        processWeek(df, 'TimeStamp', metadata[metadata['DataPackageID'] == '_'.join(course.split('.')[0].split('_')[2:])]['StartDate'].values[0])

        if config['outputType'] == 'no-proc':
            sessionDict[course] = df
            continue

        sessions = []

        for index, group in tqdm(df.groupby(['AccountUserID'])):
            group = group[~group['EventType'].str.contains('Transcript')][['TimeStamp', 'EventType']].sort_values('TimeStamp')
            group['Interval'] = (group['TimeStamp'] - group['TimeStamp'].shift(1))
            group['Interval'] = group['Interval'].apply(lambda x: x.total_seconds())
            group['SessionID'] = (group['TimeStamp'] - group['TimeStamp'].shift(1) > pd.Timedelta(config['maxSessionLength'], 'm')).cumsum() + 1
            session = group.groupby('SessionID').count()
            session['AccountUserID'] = index
            session['Event'] = group.groupby('SessionID')['EventType'].apply(','.join).values
            session['Interval'] = group.groupby('SessionID')['Interval'].apply(lambda x: list(x)[1:]).values
            session['NoEvents'] = session['TimeStamp']
            session['Start'] = group.drop_duplicates(subset=['SessionID'], keep='first')['TimeStamp'].values
            session['End'] = group.drop_duplicates(subset=['SessionID'], keep='last')['TimeStamp'].values
            session['Duration'] = session.apply(lambda row: getSeconds(row), axis=1)

            session = session[['AccountUserID', 'Start', 'End', 'Duration', 'NoEvents', 'Event', 'Interval']].reset_index()

            startDate = metadata[metadata['DataPackageID'] == '_'.join(course.split('.')[0].split('_')[2:])]['StartDate'].values[0]
            processWeek(session, 'Start', startDate)

            sessions.append(session)

        if len(sessions) > 0:
            sessions = pd.concat(sessions, ignore_index=True)
            sessions = sessions[sessions['NoEvents'] >= config['minNoActions']]

            if config['outputType'] == 'df':
                sessionDict[course] = sessions
            elif config['outputType'] == 'list-raw':
                sessionDict[course] = sessions['Event'].values
            elif config['outputType'] == 'list-with-id':
                sessionDict[course] = (sessions['AccountUserID'].values, sessions['Event'].values)
            elif config['outputType'] == 'list-with-id-time':
                sessionDict[course] = (sessions['AccountUserID'].values, sessions['Event'].values, sessions['Interval'].values)
            else:
                raise NotImplementedError('The output type ' + config['outputType'] + ' is not implemented.')
        else:
            print('no sessions for course', course)

    return sessionDict

def computeFeatures(event_data, feature_labels, weeks, weekly=False):
    feature_sets = {}

    for findex, ffunc in enumerate(feature_labels):
        print('processing', ffunc.getName(), 'features ...')
        flabel = ffunc.getName()
        feature_sets[flabel] = {}

        for windex, wid in enumerate(weeks):
            print('>> processing week', wid)
            feature_sets[flabel][wid] = []

            print('>> found weekly parameter to', weekly)
            if weekly:
                for w in range(wid):
                    print('>> processing the week', w)
                    feature_sets[flabel][wid][w] = []
                    for uindex, uid in tqdm(enumerate(event_data['AccountUserID'])):
                        udata = event_data[(event_data['AccountUserID'] == uid) & (event_data['Week'] == w)]
                        feature_sets[flabel][wid][w].append(ffunc.getUserFeatures(udata, wid) if len(udata) > 0 else [0 for i in range(ffunc.getNbFeatures())])
            else:
                for uindex, uid in tqdm(enumerate(event_data['AccountUserID'])):
                    udata = event_data[(event_data['AccountUserID'] == uid) & (event_data['Week'] < wid)]
                    feature_sets[flabel][wid].append(ffunc.getUserFeatures(udata, wid) if len(udata) > 0 else [0 for i in range(ffunc.getNbFeatures())])

    return feature_sets

def main():

    metadata = pd.read_csv('metadata.csv', encoding='iso-8859-1')

    config = {
        'courses': ['ca_courseware_EPFL-AlgebreLineaire-2019', 'ca_courseware_EPFL-Algebre1-Testerman-2018', 'ca_courseware_EPFL-Algebre2-Testerman-2018', 'ca_courseware_EPFL-Algebre3-Testerman-2018',
                    'ca_courseware_EPFL-CS-210-2018_t3', 'ca_courseware_EPFL-progfun1-2018_T1', 'ca_courseware_EPFL-progfun2-2018_T1',
                    'ca_courseware_EPFL-CS-206-2019_T1', 'ca_courseware_EPFL-parprog1-2018_T1'],
        'maxSessionLength': 120,
        'minNoActions': 3,
        'outputType': 'no-proc'
    }

    sessionDict = getSessions(config, metadata)

    '''

    featureLabels = [ChenCui()]
    for course, data in sessionDict.items():
        featureSets = computeFeatures(data, featureLabels, [2, 4])
    '''

if __name__ == '__main__':
    main()