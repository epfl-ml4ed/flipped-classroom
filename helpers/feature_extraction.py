from helpers.db_query import getTotalProblemsFlippedPeriod
from scipy.spatial.distance import jensenshannon
import numpy as np
import scipy
import pandas as pd
import json
from datetime import datetime, timedelta


""" WRAPPER FUNCTION"""

# def compute_feature(feat_func, df):
#     """
#     Compute the given feature on a dataframe containing events from a certain user
#     on a period of time.
#     The list of regularity and AIED features are respectively `regularity_features` and
#     `aied_features`.
#     """ 
#     if feat_func in regularity_features:
#         if feat_func in [NQZ, PQZ]:
#             return feat_func(df)
#         else:
#             #The regularity features takes the week span of the events and the timestamps 
#             #each events as arguments
#             T = df['TimeStamp'].sort_values() - df['TimeStamp'].min() #Make timestamps start from 0
#             # Compute the length (in weeks) of the period covered by the df 
#             # by converting the max timestamp to week since the first timestamp is 0
#             Lw = T.max() // (3600 * 24 * 7) + 1 
#             return feat_func(Lw, T) #Compute the feature given in argument
#     elif feat_func in aied_features:
#         # The AIED features take the whole dataframe in argument
#         return feat_func(df)
#     else:
#         raise ValueError('Unknow feature function:', feat_func)


""" REGULARITY FEATURES """

def df_to_params(df):
    #The regularity features takes the week span of the events and the timestamps 
    #each events as arguments
    T = df['TimeStamp'].sort_values().values - df['TimeStamp'].min() #Make timestamps start from 0
    # Compute the length (in weeks) of the period covered by the df 
    # by converting the max timestamp to week since the first timestamp is 0
    Lw = T.max() // (3600 * 24 * 7) + 1 
    return Lw, T

HOUR_TO_SECOND = 60 * 60
DAY_TO_SECOND = 24 * HOUR_TO_SECOND
WEEK_TO_SECOND = 7 * DAY_TO_SECOND

def studentActivity(W, T, x):
    T = np.floor_divide(T, W)
    return int(x in T)

def dailyActivity(Ld, T):
    def activity_at_hour(h):
        res = 0
        for i in range(Ld):
            res += studentActivity(HOUR_TO_SECOND, T, 24*i + h)
        return res
    hist = list(range(24))
    return list(map(activity_at_hour, hist))

def weeklyActivity(Lw, T):
    def activity_at_day(d):
        res = 0
        for i in range(Lw):
            res += studentActivity(DAY_TO_SECOND, T, 7*i + d)
        return res
    hist = list(range(7))
    return list(map(activity_at_day, hist))

def dayActivityByWeek(Lw, T):
    def activity_at_day(w,d):
        res = 0
        for h in range(24):
            res += studentActivity(HOUR_TO_SECOND, T, w*7*24 + d*24 + h)
        return res
    days = np.zeros((Lw, 7))
    for w in range(Lw):
        for d in range(7):
             days[w,d] = activity_at_day(w,d)
    return days

def PDH(df):
    Lw, T = df_to_params(df)
    activity = np.array(dailyActivity(Lw * 7, T))
    normalized_activity = activity / np.sum(activity) if np.sum(activity) else activity
    entropy = scipy.special.entr(normalized_activity).sum()
    # print(T)
    return (np.log2(24) - entropy) * np.max(activity)

def PWD(df):
    Lw, T = df_to_params(df)
    activity = np.array(weeklyActivity(Lw, T))
    normalized_activity = activity / np.sum(activity) if np.sum(activity) else activity
    entropy = scipy.special.entr(normalized_activity).sum()
    return (np.log2(7) - entropy) * np.max(activity)

def chi2Divergence(p1, p2, a1, a2):
    a = p1 - p2
    b = p1 + p2
    frac = np.divide(a, b, out=np.zeros(a.shape, dtype=float), where=b != 0)
    m1 = np.where(a1 == 1)[0]
    m2 = np.where(a2 == 1)[0]
    union = np.union1d(m1, m2)
    if (len(union) == 0): return np.nan
    return 1 - (1 / len(union)) * np.sum(np.square(frac))

def similarityDays(wi, wj):
    m1, m2 = np.where(wi == 1)[0], np.where(wj == 1)[0]
    if len(m1) == 0 or len(m2) == 0:
        return 0
    return len(np.intersect1d(m1, m2)) / max(len(m1), len(m2))

def WS1(df):
    Lw, T = df_to_params(df)
    hist = np.array([studentActivity(DAY_TO_SECOND, T, x_i) for x_i in range((Lw*WEEK_TO_SECOND)//(DAY_TO_SECOND))]).reshape([Lw, 7])
    return np.mean([similarityDays(hist[i], hist[j]) for i in range(Lw) for j in range(i + 1, Lw)])

def activityProfile(Lw, T):
    X =  np.array([studentActivity(HOUR_TO_SECOND, T, x_i) for x_i in range((Lw*WEEK_TO_SECOND)//(HOUR_TO_SECOND))]).reshape([Lw, 7*24])
    return [week.reshape([7, 24]).sum(axis=1) for week in X]

def WS2(df):
    Lw, T = df_to_params(df)
    profile = activityProfile(Lw, T)
    res = []
    for i in range(Lw):
        for j in range(i + 1, Lw):
            if not profile[i].any() or not profile[j].any(): continue
            res.append(1 - jensenshannon(profile[i], profile[j], 2.0))
    if len(res) == 0: return np.nan
    res = np.clip(np.nan_to_num(res), 0, 1) #Bound values between 0 and 1
    return np.mean(res)

def WS3(df):
    Lw, T = df_to_params(df)
    profile = activityProfile(Lw, T)
    hist = np.array([studentActivity(DAY_TO_SECOND, T, x_i) for x_i in range((Lw*WEEK_TO_SECOND)//(DAY_TO_SECOND))]).reshape([Lw, 7])
    res = []
    for i in range(Lw):
        for j in range(i + 1, Lw):
            if not profile[i].any() or not profile[j].any(): continue
            res.append(chi2Divergence(profile[i], profile[j], hist[i], hist[j]))
    if len(res) == 0: return np.nan
    res = np.clip(np.nan_to_num(res), 0, 1) #Bound values between 0 and 1
    return np.mean(res)

def fourierTransform(Xi, f, n):
    M = np.exp(-2j * np.pi * f * n)
    return np.dot(M, Xi)

def FDH(df, f=1/24):
    Lw, T = df_to_params(df)
    Xi =  np.array([studentActivity(HOUR_TO_SECOND, T, x_i) for x_i in range((Lw*WEEK_TO_SECOND)//(HOUR_TO_SECOND))])
    n = np.arange((Lw * 7 * 24 * 60 * 60) // (60 * 60))
    return abs(fourierTransform(Xi, f, n))

def FWH(df, f=1/(7*24)):
    Lw, T = df_to_params(df)
    Xi =  np.array([studentActivity(HOUR_TO_SECOND, T, x_i) for x_i in range((Lw*WEEK_TO_SECOND)//(HOUR_TO_SECOND))])
    n = np.arange((Lw * 7 * 24 * 60 * 60) // (60 * 60))
    return abs(fourierTransform(Xi, f, n))

def FWD(df, f=1/7):
    Lw, T = df_to_params(df)
    Xi =  np.array([studentActivity(DAY_TO_SECOND, T, x_i) for x_i in range((Lw*WEEK_TO_SECOND)//(DAY_TO_SECOND))])
    n = np.arange((Lw * 7 * 24 * 60 * 60) // (24 * 60 * 60))
    return abs(fourierTransform(Xi, f, n))

def NQZ(df):
    """
    Number of quiz completed over the semester
    Columns required: ProblemID
    """
    return df.ProblemID.nunique()

def PQZ(df):
    """
    Proportion of quiz completed over the flipped period
    Column required: ProblemID, Year
    """
    if len(df) == 0:
        return 0.0
    no_problems_per_year = df.groupby(by=["Year"]).count()
    temp = no_problems_per_year.reset_index()[['Year','ProblemID']].iloc[0]
    return temp[1] / getTotalProblemsFlippedPeriod(temp[0])

def getFirstViewings(video_df):
    """
    Filter the video events so that the returned dataframe only contains
    the first play event of each different videos viewed by the student with id studentID.
    Columns required: AccountUserID, VideoID, EventType, TimeStamp, Year
    """
    return video_df.loc[video_df.EventType == "Video.Play"]\
            .sort_values(by="TimeStamp").drop_duplicates(subset=["VideoID"], keep="first")\
            [["AccountUserID", "VideoID", "Year", "TimeStamp", "Subchapter"]]

def getFirstCompletions(problem_df):
    """
    Filter the problem (=quiz) events so that the returned dataframe only contains
    the first completion of each different quizzes done by studentID
    Columns required: AccountUserID, ProblemID, EventType, TimeStamp
    """
    return problem_df.loc[problem_df.EventType == "Problem.Check"]\
            .sort_values(by="TimeStamp").drop_duplicates(subset=["ProblemID"], keep="first")\
            [["ProblemID", "TimeStamp", "Subchapter"]]

def mergeOnSubchapter(viewing_df, completion_df, dated_videos_df, dated_problems_df):
    """
    Merge the video viewings with the quiz completion for a student.
    viewing_df columns required: VideoID, TimeStamp
    completion_df columns required: ProblemID, TimeStamp
    dated_videos_df columns required: VideoID, Subchapter
    dated_problem_df columns required: ProblemID, Subchapter
    """
    
    return viewing_df.merge(dated_videos_df[["VideoID", "Subchapter"]])\
            .merge(dated_problems_df[["ProblemID", "Subchapter"]])\
            .rename(columns={"TimeStamp":"TimeStamp_Video"})\
            .merge(completion_df)\
            .rename(columns={"TimeStamp":"TimeStamp_Quiz"})

def IQR(data):
    """
    Compute the Inter-Quartile Range of an array, that is, the difference between 
    the third quantile and the first quantile. Thus half the values 
    are located in this range.
    """
    q3, q1 = np.percentile(data, [75 ,25])
    iqr = q3 - q1
    return iqr

def IVQ(video_df, problem_df):
    """
    For every completed quiz, compute the time intervals (minutes)
    between the first viewing of the video and the quiz completion
    and return the interquartile range of the time intervals.
    video_df columns required: AccountUserID, VideoID, EventType, TimeStamp, Year
    problem_df columns required: AccountUserID, ProblemID, EventType, TimeStamp
    dated_videos_df columns required: VideoID, Subchapter
    dated_problem_df columns required: ProblemID, Subchapter 
    """
    viewing_df = getFirstViewings(video_df).rename(columns={'TimeStamp':'TimeStamp_Video'})
    completion_df = getFirstCompletions(problem_df).rename(columns={'TimeStamp':'TimeStamp_Quiz'})
    print("View",len(viewing_df))
    merged_df = viewing_df.merge(completion_df, on=['Subchapter'])
    print("Merge", len(merged_df))
    # merged_df = mergeOnSubchapter(viewing_df, completion_df, dated_videos_df, dated_problems_df)
    time_intervals = np.array(merged_df.TimeStamp_Quiz - merged_df.TimeStamp_Video)
    time_intervals = np.log(time_intervals[time_intervals > 0]) #log scale because of extreme values
    return IQR(time_intervals)

def SRQ(problem_df):
    """
    Measures the repartition of the quiz completions. The std (in hours) of the time intervals is computed
    aswell as the dates of completions. The smaller the std is, the more regular the student is.
    Columns required: AccountUserID, ProblemID, EventType, TimeStamp
    """
    completion_df = getFirstCompletions(problem_df)
    return np.diff(completion_df.TimeStamp.values).std() / 3600

regularity_features = [PDH, PWD, WS1, WS2, WS3, FDH, FWD, FWH, NQZ, PQZ] #, IVQ, SRQ]


""" AIED FEATURES """


def total_views(df):
    """ 
    Counts the total of videos views (rewatch and interruption included)
    Assumption: consider that a video is watched at most once per day
    """
    copy = df.copy()
    copy['Day'] = df.Date.dt.date
    #From the assumption the video view is a unique pair (video id, day)
    return len(copy.drop_duplicates(subset=['VideoID','Day'])) 

def week_video_total(year):
    """
    Returns a Series with week numbers as index and the number of videos to watch per week
    """
    with open('../config/linear_algebra.json') as f:
        config = json.load(f)
    year = str(year)
    weekly_count = config[year]["WeeklyVideoCount"]
    flipped_weeks = len(config[year]["FlippedWeeks"])
    start_week = int(datetime.strptime(config[year]["StartFlipped"], '%Y-%m-%d %H:%M:%S').strftime("%V")) #Get the 1st week number
    weeks = list(range(start_week, start_week + flipped_weeks))
    return pd.DataFrame(index=weeks, data=weekly_count, columns=["Total"])

def get_dated_videos():
    PATH = '../data/lin_alg_moodle/video_with_durations.csv'
    dated_videos = pd.read_csv(PATH, index_col=0)
    dated_videos['Due_date'] = pd.to_datetime(dated_videos['Due_date']) #Convert String to datetime
    dated_videos['Year'] = dated_videos.Due_date.dt.year #Add year column
    return dated_videos

def videos_watched_on_right_week(user_events):
    dated_videos = get_dated_videos()
    first_views = user_events.merge(dated_videos, on=['VideoID', 'Year'])
    first_views['From_date'] = first_views.Due_date - timedelta(weeks=1)
    return first_views[(first_views.Date >= first_views.From_date) & (first_views.Date <= first_views.Due_date)]

def weekly_prop(user_events):
    """
    Compute the ratio of video events in the dataframe over the videos assigned weekly the user_events
    may only contained only the first viewings, only rewatched videos or only interrupted videos.
    Columns required: VideoID, Year (YYYY format), Date (datetime object) 
    """
    if len(user_events) == 0:
        return np.array([0])
    first_views = videos_watched_on_right_week(user_events)
    #Freq Weekly starting on Thursday since the last due date is on Thursday
    weekly_count = first_views.groupby(pd.Grouper(key="Date", freq="W-THU")).size().to_frame(name="Count")
    #Convert dates to week number
    weekly_count.index = [int(week) for week in weekly_count.index.strftime("%V")]
    #Number of assigned videos per week
    weekly_total = week_video_total(user_events.Year.iloc[0])
    #Merge and compute the ratio of watched
    weekly_prop = weekly_total.merge(weekly_count, left_index=True, right_index=True, how='left')
    weekly_prop['Count'] = weekly_prop.Count.fillna(0) #Set nan to 0
    return np.clip((weekly_prop.Count / weekly_prop.Total).values,0,1)

# Average and SD of the proportion of videos watched per week
def weekly_prop_watched(user_events):
    """Compute the proportion of videos watched (nb of videos watched / nb of videos assigned)"""
    first_views = user_events.drop_duplicates(subset=["VideoID"]) #Only keep the first views per video
    return weekly_prop(first_views)

def avg_weekly_prop_watched(df): 
    return weekly_prop_watched(df).mean()

def std_weekly_prop_watched(df):
    return weekly_prop_watched(df).std()

# Average and SD of the proportion of videos replayed per week
def weekly_prop_replayed(user_events):
    """Compute the proportion of videos replayed (nb of videos replayed / nb of videos assigned)"""
    # We assume that a student watches a video at most once per day (cannot have multiple replays in one day)
    replayed_events = user_events.copy()
    replayed_events['Day'] = replayed_events.Date.dt.date #Create column with the date but not the time
    replayed_events.drop_duplicates(subset=['VideoID', 'Day'], inplace=True) #Only keep on event per video per day
    replayed_events = replayed_events[replayed_events.duplicated(subset=['VideoID'])] # Keep the replayed videos
    return weekly_prop(replayed_events)

def avg_weekly_prop_replayed(df): 
    return weekly_prop_replayed(df).mean()

def std_weekly_prop_replayed(df):
    return weekly_prop_replayed(df).std()

def weekly_prop_interrupted(user_events):
    STOP_EVENTS = ['Video.Pause', 'Video.Stop', 'Video.Load']
    df = user_events.copy() #Sort in descreasing order
    df.sort_values(by="TimeStamp", inplace=True)
    df['Diff'] = abs(df.TimeStamp.diff(-1))
    df['NextVideoID'] = df.VideoID.shift(-1)
    
    df = df[(df.Duration - df.CurrentTime > 60)] #Remove events in the last minute
    # Interruption when
    #   * a break is too long
    #   * a break (not in the last minute) is followed by an event in another video (the user left the video)
    #   * an event occurs in another video before the end of the current video
    break_too_long = (df.EventType.isin(STOP_EVENTS)) & (df.Diff > 3600)
    break_then_other_video = (df.EventType.isin(STOP_EVENTS)) &  (df.VideoID != df.NextVideoID)
    event_other_video = (df.VideoID != df.NextVideoID) & ((df.Duration - df.CurrentTime) > (df.Diff))
    interrup_conditions = break_too_long | break_then_other_video | event_other_video
    df = df[interrup_conditions]
    return weekly_prop(df)

def avg_weekly_prop_interrupted(df): 
    return weekly_prop_interrupted(df).mean()

def std_weekly_prop_interrupted(df):
    return weekly_prop_interrupted(df).std() 

def total_actions(user_events):
    """Counts the total number of actions performed across every videos"""
    return len(user_events)

def frequency_all_actions(user_events):
    """Compute the frequency of actions performed per hour spent watching videos"""
    user_events = user_events.copy()
    user_events.loc[:,'Day'] = user_events.loc[:,'Date'].dt.date #Create column with the date but not the time
    user_events.drop_duplicates(subset=['VideoID', 'Day'], inplace=True) #Only keep on event per video per day
    watching_time = user_events.Duration.sum() / 3600 # hours
    return total_actions(user_events) / watching_time if watching_time != 0 else 0

def count_actions(user_events, action):
    """Count the total number of events with type `action`"""
    if 'Backward' in action:
        user_events = user_events[(user_events.EventType == 'Video.Seek') & 
                                  (user_events.OldTime < user_events.NewTime)]
    elif 'Forward' in action:
        user_events = user_events[(user_events.EventType == 'Video.Seek') & 
                                  (user_events.OldTime > user_events.NewTime)]
    else:
        user_events = user_events[user_events.EventType == action]        
    return len(user_events)

def freq_play(user_events):
    return count_actions(user_events,'Video.Play') / total_actions(user_events)

def freq_pause(user_events):
    return count_actions(user_events,'Video.Pause') / total_actions(user_events)

def freq_seek_backward(user_events):
    return count_actions(user_events,'Video.SeekBackward') / total_actions(user_events)

def freq_seek_forward(user_events):
    return count_actions(user_events,'Video.SeekForward') / total_actions(user_events)

def freq_speed_change(user_events):
    return count_actions(user_events,'Video.SpeedChange') / total_actions(user_events)

def freq_stop(user_events):
    return count_actions(user_events,'Video.Stop') / total_actions(user_events)

def pause_duration(user_events, max_duration=500):
    """
    Compute the time interval between each pause event and the next play event`
    Only pause durataions smaller than `max_duration` are taken into account. 
    Default threshold to 10 min
    """
    pause_events = user_events[user_events.EventType.isin(["Video.Pause", "Video.Play"])].copy()
    pause_events = pause_events.sort_values(by="TimeStamp")
    pause_events['PrevEvent'] = pause_events['EventType'].shift(1)
    pause_events['Diff'] = pause_events.TimeStamp.diff().dropna()
    pause_events = pause_events[pause_events.PrevEvent == 'Video.Pause']
    nb_pause = len(pause_events)
    pause_events = pause_events[pause_events.Diff < max_duration]
    return pause_events.Diff.values

def avg_pause_duration(user_events):
    return pause_duration(user_events).mean()

def std_pause_duration(user_events):
    return pause_duration(user_events).std()

def seek_length(user_events):
    user_events = user_events[user_events.EventType == 'Video.Seek']
    return abs(user_events.OldTime - user_events.NewTime).values

def avg_seek_length(user_events):
    return seek_length(user_events).mean()

def std_seek_length(user_events):
    return seek_length(user_events).std()

def compute_speedchange_current_time(user_events):
    """Compute the CurrentTime of the SpeedChange event as it is not logged in the db
    For that we find the closest event (in the same video and day) with a non null CurrentTime
    Then we compute the SpeedChange CurrentTime with the delta time between the 2 events
    Columns required: VideoID, Timestamp, EventType, CurrentTime, Duration
    """
    #Keep only SpeedChange events and events with non null CurrentTime
    df = user_events[(user_events.EventType == 'Video.SpeedChange') | (~user_events.CurrentTime.isna())]
    #Compute the deltatime between the previous events
    df = df.sort_values(by="TimeStamp", ascending=True)
    df['NextDiff'] = abs(df.TimeStamp.diff())
    df['PrevDiff'] = abs(df.TimeStamp.diff(-1))

    ### Define the closest event information: VideoID, TimeStamp and CurrentTime ###
    df['NextVideoID'] = df.VideoID.shift()
    df['NextTimeStamp'] = df.TimeStamp.shift()
    df['NextCurrentTime'] = df.CurrentTime.shift()
    df['PrevVideoID'] = df.VideoID.shift(-1)
    df['PrevTimeStamp'] = df.TimeStamp.shift(-1)
    df['PrevCurrentTime'] = df.CurrentTime.shift(-1)

    df['ClosestVideoID'] = np.nan
    df['ClosestTimeStamp'] = np.nan
    df['ClosestCurrentTime'] = np.nan
    #If the next event is the closest
    df.loc[df.NextDiff <= df.PrevDiff, 'ClosestVideoID'] = df[df.NextDiff <= df.PrevDiff].NextVideoID
    df.loc[df.NextDiff <= df.PrevDiff, 'ClosestTimeStamp'] = df[df.NextDiff <= df.PrevDiff].NextTimeStamp
    df.loc[df.NextDiff <= df.PrevDiff, 'ClosestCurrentTime'] = df[df.NextDiff <= df.PrevDiff].NextCurrentTime
    #If the previous event is the closest
    df.loc[df.NextDiff > df.PrevDiff, 'ClosestVideoID'] = df[df.NextDiff > df.PrevDiff].PrevVideoID
    df.loc[df.NextDiff > df.PrevDiff, 'ClosestTimeStamp'] = df[df.NextDiff > df.PrevDiff].PrevTimeStamp
    df.loc[df.NextDiff > df.PrevDiff, 'ClosestCurrentTime'] = df[df.NextDiff > df.PrevDiff].PrevCurrentTime

    #Filter the SpeedChange events
    df = df[(df.EventType == 'Video.SpeedChange') & (df.VideoID == df.ClosestVideoID) &
               (~df.ClosestCurrentTime.isna())]
    df['CurrentTime'] = df.ClosestCurrentTime + abs(df.TimeStamp - df.ClosestTimeStamp)

    df = df[df.CurrentTime < df.Duration]
    return df

def compute_time_speeding_up(user_events):
    """
    Compute the time spent with a high speed (> 1) for each video
    Columns required: VideoID, Timestamp, EventType, CurrentTime, Duration
    """
    df = user_events.copy()
    sc = compute_speedchange_current_time(user_events)[['TimeStamp', 'EventType', 'CurrentTime']]
    df = df.merge(sc, on=['TimeStamp', 'EventType'], how='left')
    df['CurrentTime'] = np.where(df.CurrentTime_y.isna(), df.CurrentTime_x, df.CurrentTime_y)
    df.drop(columns=['CurrentTime_x', 'CurrentTime_y'], inplace=True)
    df = df.sort_values(by='TimeStamp')

    def label_speed(row, speed):
        newSpeed = speed['value'] if np.isnan(row) else row
        speed['value'] = newSpeed
        return newSpeed

    # Create a mutable object instead of a simple integer in order to modify 
    # its value in the label_speed function
    speed = {"value": 1}
    df['Speed'] = df.NewSpeed.apply(lambda row: label_speed(row, speed))
    df['NextVideoID'] = df.VideoID.shift(-1)
    df['SpeedUpTime'] = abs(df.TimeStamp.diff(-1))

    #If the SpeedUpTime is Nan (last elem of DataFrame) or if the next events is in another video
    #then the time speeding up is until the end of the video
    conditions = (df.SpeedUpTime.isna()) | (df.VideoID != df.NextVideoID)
    df['SpeedUpTime'] = np.where(conditions, df.Duration - df.CurrentTime, df.SpeedUpTime)

    #When the speed is switched back to normal or slower then we stop counting
    conditions = (df.Speed <= 1) | (df.EventType.isin(['Video.Stop', 'Video.Pause'])) | \
                (df.SpeedUpTime > 3600)
    df['SpeedUpTime'] = np.where(conditions, 0, df.SpeedUpTime)
    return df.groupby('VideoID').SpeedUpTime.sum().values

def avg_time_speeding_up(user_events):
    return compute_time_speeding_up(user_events).mean()

def std_time_speeding_up(user_events):
    return compute_time_speeding_up(user_events).std()

aied_features = [total_views, avg_weekly_prop_watched, std_weekly_prop_watched, avg_weekly_prop_replayed,
    std_weekly_prop_replayed, avg_weekly_prop_interrupted, std_weekly_prop_interrupted, total_actions,
    frequency_all_actions, freq_play, freq_pause, freq_seek_backward, freq_seek_forward, freq_speed_change,
    freq_stop, avg_pause_duration, std_pause_duration, avg_seek_length, std_seek_length, avg_time_speeding_up,
    std_time_speeding_up]