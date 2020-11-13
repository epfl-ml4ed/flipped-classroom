from helpers.db_query import getTotalProblemsFlippedPeriod
from scipy.spatial.distance import jensenshannon
import numpy as np
import scipy

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

def PDH(Lw, T):
    activity = np.array(dailyActivity(Lw * 7, T))
    normalized_activity = activity / np.sum(activity) if np.sum(activity) else activity
    entropy = scipy.special.entr(normalized_activity).sum()
    return (np.log2(24) - entropy) * np.max(activity)

def PWD(Lw, T):
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

def WS1(Lw, T):
    hist = np.array([studentActivity(DAY_TO_SECOND, T, x_i) for x_i in range((Lw*WEEK_TO_SECOND)//(DAY_TO_SECOND))]).reshape([Lw, 7])
    return np.mean([similarityDays(hist[i], hist[j]) for i in range(Lw) for j in range(i + 1, Lw)])

def activityProfile(Lw, T):
    X =  np.array([studentActivity(HOUR_TO_SECOND, T, x_i) for x_i in range((Lw*WEEK_TO_SECOND)//(HOUR_TO_SECOND))]).reshape([Lw, 7*24])
    return [week.reshape([7, 24]).sum(axis=1) for week in X]

def WS2(Lw, T):
    profile = activityProfile(Lw, T)
    res = []
    for i in range(Lw):
        for j in range(i + 1, Lw):
            if not profile[i].any() or not profile[j].any(): continue
            res.append(1 - jensenshannon(profile[i], profile[j], 2.0))
    if len(res) == 0: return np.nan
    res = np.clip(np.nan_to_num(res), 0, 1) #Bound values between 0 and 1
    return np.mean(res)

def WS3(Lw, T):
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

def FDH(Lw, T, f = 1/24):
    Xi =  np.array([studentActivity(HOUR_TO_SECOND, T, x_i) for x_i in range((Lw*WEEK_TO_SECOND)//(HOUR_TO_SECOND))])
    n = np.arange((Lw * 7 * 24 * 60 * 60) // (60 * 60))
    return abs(fourierTransform(Xi, f, n))

def FWH(Lw, T, f=1/(7*24)):
    Xi =  np.array([studentActivity(HOUR_TO_SECOND, T, x_i) for x_i in range((Lw*WEEK_TO_SECOND)//(HOUR_TO_SECOND))])
    n = np.arange((Lw * 7 * 24 * 60 * 60) // (60 * 60))
    return abs(fourierTransform(Xi, f, n))

def FWD(Lw, T, f=1/7):
    Xi =  np.array([studentActivity(DAY_TO_SECOND, T, x_i) for x_i in range((Lw*WEEK_TO_SECOND)//(DAY_TO_SECOND))])
    n = np.arange((Lw * 7 * 24 * 60 * 60) // (24 * 60 * 60))
    return abs(fourierTransform(Xi, f, n))

def NQZ(Pw, id):
    no_problems_per_student = Pw[['AccountUserID','ProblemID']].groupby('AccountUserID').count()
    return no_problems_per_student.loc[str(id)][0] if str(id) in no_problems_per_student.index else 0.0

def PQZ(Pw, id):
    no_problems_per_year = Pw.groupby(by=["AccountUserID", "Year"]).count()
    if str(id) not in no_problems_per_year.index:
        return 0.0
    temp = no_problems_per_year.loc[str(id)].reset_index().iloc[0]
    return temp[1] / getTotalProblemsFlippedPeriod(temp[0])

def getFirstViewings(video_df, sid):
    """
    Filter the video events so that the returned dataframe only contains
    the first play event of each different videos viewed by the student with id studentID.
    """
    return video_df.loc[(video_df["AccountUserID"] == str(sid)) & (video_df.EventType == "Video.Play")]\
            .sort_values(by="TimeStamp").drop_duplicates(subset=["VideoID"], keep="first")\
            [["AccountUserID", "VideoID", "Year", "TimeStamp"]]

def getFirstCompletions(problem_df, sid):
    """
    Filter the problem (=quiz) events so that the returned dataframe only contains
    the first completion of each different quizzes done by studentID
    """
    return problem_df.loc[(problem_df["AccountUserID"] == str(sid)) & (problem_df.EventType == "Problem.Check")]\
            .sort_values(by="TimeStamp").drop_duplicates(subset=["ProblemID"], keep="first")\
            [["ProblemID", "TimeStamp"]]

def mergeOnSubchapter(viewing_df, completion_df, dated_videos_df, dated_problems_df):
    """
    Merge the video viewings with the quiz completion for a student.
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

def IVQ(sid, video_df, problem_df, dated_videos_df, dated_problems_df):
    """
    For every completed quiz, compute the time intervals (minutes)
    between the first viewing of the video and the quiz completion
    and return the interquartile range of the time intervals
    """
    viewing_df = getFirstViewings(video_df, sid)
    completion_df = getFirstCompletions(problem_df, sid)
    merged_df = mergeOnSubchapter(viewing_df, completion_df, dated_videos_df, dated_problems_df)
    time_intervals = np.array(merged_df.TimeStamp_Quiz - merged_df.TimeStamp_Video)
    time_intervals = np.log(time_intervals[time_intervals > 0]) #log scale because of extreme values
    return IQR(time_intervals)

def SRQ(sid, problem_df):
    """
    Measures the repartition of the quiz completions. The std (in hours) of the time intervals is computed
    aswell as the dates of completions. The smaller the std is, the more regular the student is.
    """
    completion_df = getFirstCompletions(problem_df, sid)
    return np.diff(completion_df.TimeStamp.values).std() / 3600

def compute_feature(feat_func, df):
    """Compute the given feature (PDH, WS1, FDH, etc.) on the given dataframe."""
    T = df['TimeStamp'].sort_values() - df['TimeStamp'].min() #Make timestamps start from 0
    # Compute the length (in weeks) of the period covered by the df 
    # by converting the max timestamp to week since the first timestamp is 0
    Lw = T.max() // (3600 * 24 * 7) + 1 
    return feat_func(Lw, T) #Compute the feature given in argument