from datetime import date
import pandas as pd
import numpy as np
import json
import time

def getRepeatingStudentsIDs(df):
    repentants = []
    for s in set(df['AccountUserID']):
        x = df[df['AccountUserID'] == s]
        if len(x[x['Year'] == 2017]) > 0 and len(x[x['Year'] == 2018]) > 0 or len(x[x['Year'] == 2018]) > 0 and len(x[x['Year'] == 2019]) > 0:
            repentants.append(s)
    return repentants

def getUnactiveStudentsIDs(df):
    vid_cnt = df[['AccountUserID', 'VideoID']].drop_duplicates().groupby('AccountUserID').count().reset_index().rename(columns={'VideoID': 'Count'})
    return list(vid_cnt[vid_cnt['Count'] <= 60]['Count'].values)

def getStudentTimeStamps(df, id):
    with open('../config/linear_algebra.json') as f:
        config = json.load(f)
    sid = str(list(df['AccountUserID'].sample(1))[0] if not id else id)
    sdata = df[df['AccountUserID'] == sid]
    sdate = config[str(list(sdata['Year'])[0])]['StartFlipped']
    dft = sdata['TimeStamp'].sort_values() - time.mktime(date(int(sdate.split('-')[0]), int(sdate.split('-')[1]), int(sdate.split('-')[2])).timetuple())
    return sid, list(np.where(dft < 0, 0, dft)), len(config[str(list(sdata['Year'])[0])]['FlippedWeeks'])

def transitionEvents(sessions, eventMap):
    transitionMatrix = np.zeros((len(set(eventMap.keys())), len(set(eventMap.keys()))))
    for events in sessions['Event']:
        listEvents = events.split(',')
        for i in range(len(listEvents) - 1):
            transitionMatrix[eventMap[listEvents[i]], eventMap[listEvents[i + 1]]] += 1
    return transitionMatrix

def transitionIntervals(sessions, eventMap):
    transitionMatrix = np.zeros((len(set(eventMap.keys())), len(set(eventMap.keys()))))
    for events, intervals in zip(sessions['Event'], sessions['Interval']):
        listEvents = events.split(',')
        listIntervals = intervals
        for i in range(len(listEvents) - 1):
            prev_value = transitionMatrix[eventMap[listEvents[i]], eventMap[listEvents[i + 1]]]
            next_value = listIntervals[i] if prev_value == 0 else (prev_value + listIntervals[i]) / 2
            transitionMatrix[eventMap[listEvents[i]], eventMap[listEvents[i + 1]]] = next_value
    return transitionMatrix

def getSessions(df, maxSessionLength=120, minNoActions=3):
    sessions = []
    for index, group in tqdm(df.groupby(['AccountUserID'])):
        group = group[~group['EventType'].str.contains('Transcript')][['TimeStamp', 'EventType', 'Round']].sort_values('TimeStamp')
        group['Interval'] = (group['TimeStamp'] - group['TimeStamp'].shift(1))
        group['Interval'] = group['Interval'].apply(lambda x: x.total_seconds())
        group['SessionID'] = (group['TimeStamp'] - group['TimeStamp'].shift(1) > pd.Timedelta(maxSessionLength, 'm')).cumsum() + 1
        session = group.groupby('SessionID').count()
        session['NoEvents'] = session['TimeStamp']
        session['Round'] = group.drop_duplicates(subset=['SessionID'], keep='first')['Round'].values
        session['Start'] = group.drop_duplicates(subset=['SessionID'], keep='first')['TimeStamp'].values
        session['End'] = group.drop_duplicates(subset=['SessionID'], keep='last')['TimeStamp'].values
        session['Duration'] = session.apply(lambda row: getSeconds(row), axis=1)
        session['AccountUserID'] = index
        session['Event'] = group.groupby('SessionID')['EventType'].apply(','.join).values
        session['Interval'] = group.groupby('SessionID')['Interval'].apply(lambda x: list(x)[1:]).values
        session = session[['AccountUserID', 'Round', 'Start', 'End', 'Duration', 'NoEvents', 'Event', 'Interval']].reset_index()
        sessions.append(session)
    sessions = pd.concat(sessions, ignore_index=True)
    sessions = sessions[sessions['NoEvents'] >= minNoActions]
    return sessions

def getStudentWeeklyEventTransitions(sessions, year, eventMap):
    with open('../config/linear_algebra.json') as f:
        config = json.load(f)
    eventLabels = list(eventMap.keys())
    sessions['Week'] = getCourseWeek(sessions['Start'], config[year]['Start'], config[year]['End'])
    sessions = sessions.dropna().sort_values(by='Start').copy()
    user_length_by_week = np.zeros((19, len(eventLabels), len(eventLabels)))
    for index, group in sessions.groupby('Week'):
        for events, intervals in zip(group['Event'], group['Interval']):
            listEvents = events.split(',')
            listIntervals = intervals
            for i in range(len(listEvents) - 1):
                user_length_by_week[int(index), eventMap[listEvents[i]], eventMap[listEvents[i + 1]]] += 1
        user_length_by_week[int(index)] /= np.sum(user_length_by_week[int(index)])
    return user_length_by_week

def getStudentWeeklyEventIntervals(sessions, year, eventMap):
    with open('../config/linear_algebra.json') as f:
        config = json.load(f)
    eventLabels = list(eventMap.keys())
    sessions['Week'] = getCourseWeek(sessions['Start'], config[year]['Start'], config[year]['End'])
    sessions = sessions.dropna().sort_values(by='Start').copy()
    user_length_by_week = np.zeros((19, len(eventLabels), len(eventLabels)))
    for index, group in sessions.groupby('Week'):
        for events, intervals in zip(group['Event'], group['Interval']):
            listEvents = events.split(',')
            listIntervals = intervals
            for i in range(len(listEvents) - 1):
                prev_value = user_length_by_week[int(index), eventMap[listEvents[i]], eventMap[listEvents[i + 1]]]
                next_value = listIntervals[i] if prev_value == 0 else (prev_value + listIntervals[i]) / 2
                user_length_by_week[int(index), eventMap[listEvents[i]], eventMap[listEvents[i + 1]]] = next_value
        user_length_by_week[int(index)] /= np.sum(user_length_by_week[int(index)])
    return user_length_by_week

def getStudentWeeklySessionLength(sessions, year):
    with open('../config/linear_algebra.json') as f:
        config = json.load(f)
    sessions['Week'] = getCourseWeek(sessions['Start'], config[year]['Start'], config[year]['End'])
    sessions = sessions.dropna().sort_values(by='Start').copy()
    user_length_by_week = np.zeros(19)
    t = sessions.groupby('Week')['Duration'].mean()
    for j, v in zip(list(t.index), list(t.values)):
        if int(j) < 19:
            user_length_by_week[int(j)] = v
    return user_length_by_week

def getStudentWeeklyEvents(sessions, year, eventMap):
    with open('../config/linear_algebra.json') as f:
        config = json.load(f)
    eventLabels = list(eventMap.keys())
    sessions['Week'] = getCourseWeek(sessions['Start'], config[year]['Start'], config[year]['End'])
    sessions = sessions.dropna().sort_values(by='Start').copy()
    user_length_by_week = np.zeros((19, len(eventLabels)))
    for index, group in sessions.groupby('Week'):
        for eventList in group['Event']:
            for event in eventList.split(','):
                user_length_by_week[int(index), int(eventMap[event])] += 1
        user_length_by_week[int(index)] /= np.sum(user_length_by_week[int(index)])
    return user_length_by_week