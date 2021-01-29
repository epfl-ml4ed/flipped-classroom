#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import timedelta

from helper.htime import *

def get_sessions(df, sess_length=120, min_actions=3):
    sessions = []
    for index, group in df.groupby(['user_id']):
        group = group[~group['event_type'].str.contains('Transcript')][['date', 'event_type']].sort_values('date')
        group['interval'] = (group['date'] - group['date'].shift(1))
        group['interval'] = group['interval'].apply(lambda x: x.total_seconds())
        group['session_id'] = (group['date'] - group['date'].shift(1) > pd.Timedelta(sess_length, 'm')).cumsum() + 1
        session = group.groupby('session_id').count()
        session['no_events'] = session['date']
        session['start'] = group.drop_duplicates(subset=['session_id'], keep='first')['date'].values
        session['end'] = group.drop_duplicates(subset=['session_id'], keep='last')['date'].values
        session['duration'] = session.apply(lambda row: get_seconds(row), axis=1)
        session['user_id'] = index
        session['event'] = group.groupby('session_id')['event_type'].apply(','.join).values
        session['interval'] = group.groupby('session_id')['interval'].apply(lambda x: list(x)[1:]).values
        session = session[['user_id', 'start', 'end', 'duration', 'no_events', 'event', 'interval']].reset_index()
        sessions.append(session)
    sessions = pd.concat(sessions, ignore_index=True)
    sessions = sessions[sessions['no_events'] >= min_actions]
    return sessions

def get_sequence_from_course(course, seq_length=300):
    data = course.get_clickstream()

    users = course.get_students()
    weeks = np.arange(course.get_weeks())

    maps = {v:i for i, v in enumerate(data['event_type'].unique())}
    acts = np.zeros((len(users), len(weeks), seq_length))
    tims = np.zeros((len(users), len(weeks), seq_length))

    for sid, user_id in enumerate(users):
        data = data[data['user_id'] == user_id].sort_values(by='date').copy()
        data['event_type_id'] = data['event_type'].apply(lambda x: maps[x])
        for wid in weeks:
            data_week = data[data['week'] == wid]
            if len(data_week) > 0:
                acts[sid, wid] = data_week['event_type_id'].values[:seq_length] if len(data_week) > seq_length else np.pad(data_week['event_type_id'].values, (0, seq_length - len(data_week)), 'constant')
                tims[sid, wid] = data_week['timestamp'].values[:seq_length] if len(data_week) > seq_length else np.pad(data_week['timestamp'].values, (0, seq_length - len(data_week)), 'constant')

    return acts, tims, maps

def get_speedchange_current_time(user_events):
    df = user_events[(user_events['event_type'] == 'Video.SpeedChange') | (~user_events['current_time'].isna())]

    # Compute the deltatime between the previous events
    df = df.sort_values(by='date', ascending=True)
    df['next_diff'] = abs(df['date'].diff())
    df['prev_diff'] = abs(df['date'].diff(-1))

    # Define the closest event information: VideoID, TimeStamp and CurrentTime ###
    df['next_video_id'] = df['video_id'].shift()
    df['next_date'] = df['date'].shift()
    df['next_current_time'] = df['current_time'].shift()
    df['prev_video_id'] = df['video_id'].shift(-1)
    df['prev_date'] = df['date'].shift(-1)
    df['prev_current_time'] = df['current_time'].shift(-1)

    df['closest_video_id'] = np.nan
    df['closest_date'] = np.nan
    df['closest_current_time'] = np.nan

    # If the next event is the closest
    df.loc[df['next_diff'] <= df['prev_diff'], 'closest_video_id'] = df[df['next_diff'] <= df['prev_diff']]['next_video_id']
    df.loc[df['next_diff'] <= df['prev_diff'], 'closest_date'] = df[df['next_diff'] <= df['prev_diff']]['next_date']
    df.loc[df['next_diff'] <= df['prev_diff'], 'closest_current_time'] = df[df['next_diff'] <= df['prev_diff']]['next_current_time']
    # If the previous event is the closest
    df.loc[df['next_diff'] > df['prev_diff'], 'closest_video_id'] = df[df['next_diff'] > df['prev_diff']]['prev_video_id']
    df.loc[df['next_diff'] > df['prev_diff'], 'closest_date'] = df[df['next_diff'] > df['prev_diff']]['prev_date']
    df.loc[df['next_diff'] > df['prev_diff'], 'closest_current_time'] = df[df['next_diff'] > df['prev_diff']]['prev_current_time']

    # Filter the SpeedChange events
    df = df[(df['event_type'] == 'Video.SpeedChange') & (df['video_id'] == df['closest_video_id']) & (~df['closest_current_time'].isna())]
    if len(df.index) > 0:
        df['current_time'] = df['closest_current_time'] + abs(df['date'] - df['closest_date'])
        df = df[df['current_time'] < df['duration']]
    return df

def get_time_speeding_up(df):
    udata = df.copy()
    sc = get_speedchange_current_time(udata)[['date', 'event_type', 'current_time']]
    df = df.merge(sc, on=['date', 'event_type'], how='left')
    df['current_time'] = np.where(df['current_time_y'].isna(), df['current_time_x'], df['current_time_y'])
    df.drop(columns=['current_time_x', 'current_time_y'], inplace=True)
    df = df.sort_values(by='date')

    def label_speed(row, speed):
        newSpeed = speed['value'] if np.isnan(row) else row
        speed['value'] = newSpeed
        return newSpeed

    # Create a mutable object instead of a simple integer in order to modify  its value in the label_speed function
    speed = {'value': 1}
    df['speed'] = df['new_speed'].apply(lambda row: label_speed(row, speed))
    df['next_video_id'] = df['video_id'].shift(-1)
    df['speed_up_time'] = abs(df['date'].diff(-1))

    # If the SpeedUpTime is Nan (last elem of DataFrame) or if the next events is in another video then the time speeding up is until the end of the video
    conditions = (df['speed_up_time'].isna()) | (df['video_id'] != df['next_video_id'])
    df['speed_up_time'] = np.where(conditions, df['duration'].values - df['current_time'].values, df['speed_up_time'])

    # When the speed is switched back to normal or slower then we stop counting
    conditions = (df['speed'] <= 1) | (df['event_type'].isin(['Video.Stop', 'Video.Pause'])) | (df['speed_up_time'] > 3600)
    df['speed_up_time'] = np.where(conditions, 0, df['speed_up_time'])
    return df.groupby('video_id')['speed_up_time'].sum().values

def get_seek_length(df):
    df = df[df['event_type'] == 'Video.Seek']
    return abs(df['old_time'] - df['new_time']).values

def count_actions(df, action):
    if 'Backward' in action:
        df = df[(df['event_type'] == 'Video.Seek') & (df['old_time'] > df['new_time'])]
    elif 'Forward' in action:
        df = df[(df['event_type'] == 'Video.Seek') & (df['old_time'] < df['new_time'])]
    else:
        df = df[df['event_type'] == action]
    return len(df.index)

def get_videos_watched_on_right_week(df, settings):
    first_views = df.merge(settings['course'].get_schedule()[['id', 'duration', 'date']], left_on=['video_id'], right_on=['id'])
    first_views['from_date'] = first_views['date_y'] - timedelta(weeks=1)
    return first_views[(first_views['date_x'] >= first_views['from_date']) & (first_views['date_x'] <= first_views['date_y'])]

def get_week_video_total(settings):
    return settings['course'].get_schedule().groupby(by='week').size().to_frame(name="Total")

def get_weekly_prop(df, settings):
    if len(df) == 0:
        return np.array([0])
    first_views = get_videos_watched_on_right_week(df, settings)
    # Freq Weekly starting on Thursday since the last due date is on Thursday
    weekly_count = first_views.groupby(by='week').size().to_frame(name="Count")
    # Number of assigned videos per week
    weekly_total = get_week_video_total(settings)
    # Merge and compute the ratio of watched
    weekly_prop = weekly_total.merge(weekly_count, left_index=True, right_index=True, how='left')
    weekly_prop['Count'] = weekly_prop['Count'].fillna(0) # Set nan to 0
    return np.clip((weekly_prop.Count / weekly_prop.Total).values,0,1)

def get_weekly_prop_watched(df, settings):
    return get_weekly_prop(df.drop_duplicates(subset=['video_id']), settings)

def get_weekly_prop_replayed(df, settings):
    replayed_events = df.copy()
    replayed_events['day'] = replayed_events['date'].dt.date # Create column with the date but not the time
    replayed_events.drop_duplicates(subset=['video_id', 'day'], inplace=True) # Only keep on event per video per day
    replayed_events = replayed_events[replayed_events.duplicated(subset=['video_id'])] # Keep the replayed videos
    return get_weekly_prop(replayed_events, settings)

def get_weekly_prop_interrupted(df, settings):
    STOP_EVENTS = ['Video.Pause', 'Video.Stop', 'Video.Load']
    df = df.sort_values(by='date').copy()
    df['time_diff'] = abs(df['date'].diff(-1).dt.total_seconds())
    df['next_video_id'] = df['video_id'].shift(-1)
    df = df.merge(settings['course'].get_schedule()[['id', 'duration']], left_on=['video_id'], right_on=['id'])
    df = df[(df['duration'] - df['current_time'] > 60)] # Remove events in the last minute
    break_too_long = (df['event_type'].isin(STOP_EVENTS)) & (df['time_diff'] > 3600)
    break_then_other_video = (df['event_type'].isin(STOP_EVENTS)) &  (df['video_id'] != df['next_video_id'])
    event_other_video = (df['video_id'] != df['next_video_id']) & ((df['duration'] - df['current_time']) > (df['time_diff']))
    interrup_conditions = break_too_long | break_then_other_video | event_other_video
    df = df[interrup_conditions]
    del df['duration']
    return get_weekly_prop(df, settings)
