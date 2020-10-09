from datetime import date
import pandas as pd
import numpy as np
import json
import time

def delRepentants(df):
    repentants = []
    for s in set(df['AccountUserID']):
        x = df[df['AccountUserID'] == s]
        if len(x[x['Year'] == 2017]) > 0 and len(x[x['Year'] == 2018]) > 0:
            repentants.append(s)
    return df[~df['AccountUserID'].isin(repentants)]

def delLessActive(df):
    vid_cnt = df[['AccountUserID', 'VideoID']].drop_duplicates().groupby('AccountUserID').count().reset_index().rename(columns={'VideoID': 'Count'})
    return pd.merge(left=vid_cnt[vid_cnt['Count'] > 60], right=df, on='AccountUserID').drop_duplicates()

def getStudentTimeStamps(df, id):
    with open('../config/cf_mooc.json') as f:
        config = json.load(f)
    sid = str(list(df['AccountUserID'].sample(1))[0] if not id else id)
    sdata = df[df['AccountUserID'] == sid]
    sdate = config[str(list(sdata['Year'])[0])]['Start']
    dft = sdata['TimeStamp'].sort_values() - time.mktime(date(int(sdate.split('-')[0]), int(sdate.split('-')[1]), int(sdate.split('-')[2])).timetuple())
    return sid, list(np.where(dft < 0, 0, dft)), config[str(list(sdata['Year'])[0])]['FlippedWeeks']