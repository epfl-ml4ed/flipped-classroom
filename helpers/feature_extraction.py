from helpers.db_query import getTotalProblemsFlippedPeriod
from scipy.spatial.distance import jensenshannon
import numpy as np
import scipy

def studentActivity(W, T, x):
    T = np.floor_divide(T, W)
    return int(x in T)

def dailyActivity(Ld, T):
    def activity_at_hour(h):
        res = 0
        for i in range(Ld):
            res += studentActivity(60*60, T, 24*i + h)
        return res
    hist = list(range(24))
    return list(map(activity_at_hour, hist))

def weeklyActivity(Lw, T):
    def activity_at_day(d):
        res = 0
        for i in range(Lw):
            res += studentActivity(24*60*60, T, 7*i + d)
        return res
    hist = list(range(7))
    return list(map(activity_at_day, hist))

def dayActivityByWeek(Lw, T):
    def activity_at_day(w,d):
        res = 0
        for h in range(24):
            res += studentActivity(60*60, T, w*7*24 + d*24 + h)
        return res
    days = np.zeros((Lw, 7))
    for w in range(Lw):
        for d in range(7):
             days[w,d] = activity_at_day(w,d)
    return days

def PDH(Lw, T):
    activity = np.array(dailyActivity(Lw * 7, T))
    normalized_activity = activity / np.sum(activity)
    entropy = scipy.special.entr(normalized_activity).sum()
    return (np.log2(24) - entropy) * np.max(activity)

def PWD(Lw, T):
    activity = np.array(weeklyActivity(Lw, T))
    normalized_activity = activity / np.sum(activity)
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
    hist = np.array([studentActivity(24*60*60, T, x_i) for x_i in range((Lw*7*24*60*60)//(24*60*60))]).reshape([Lw, 7])
    return np.mean([similarityDays(hist[i], hist[j]) for i in range(Lw) for j in range(i + 1, Lw)])

def activity_profile(Lw, T):
    X =  np.array([studentActivity(60*60, T, x_i) for x_i in range((Lw*7*24*60*60)//(60*60))]).reshape([Lw, 7*24])
    return [week.reshape([7, 24]).sum(axis=1) for week in X]

def WS2(Lw, T):
    profile = activity_profile(Lw, T)
    res = []
    for i in range(Lw):
        for j in range(i + 1, Lw):
            if not profile[i].any() or not profile[j].any(): continue
            res.append(1 - jensenshannon(profile[i], profile[j], 2.0))
    if len(res) == 0: return np.nan
    return np.mean(res)

def WS3(Lw, T):
    profile = activity_profile(Lw, T)
    hist = np.array([studentActivity(24*60*60, T, x_i) for x_i in range((Lw*7*24*60*60)//(24*60*60))]).reshape([Lw, 7])
    res = []
    for i in range(Lw):
        for j in range(i + 1, Lw):
            if not profile[i].any() or not profile[j].any(): continue
            res.append(chi2Divergence(profile[i], profile[j], hist[i], hist[j]))
    if len(res) == 0: return np.nan
    return np.mean(res)

def fourier_transform(Xi, f, n):
    M = np.exp(-2j * np.pi * f * n)
    return np.dot(M, Xi)

def FDH(Lw, T, f = 1/24):
    Xi =  np.array([studentActivity(60*60, T, x_i) for x_i in range((Lw*7*24*60*60)//(60*60))])
    n = np.arange((Lw * 7 * 24 * 60 * 60) // (60 * 60))
    return abs(fourier_transform(Xi, f, n))

def FWH(Lw, T, f=1/(7*24)):
    Xi =  np.array([studentActivity(60*60, T, x_i) for x_i in range((Lw*7*24*60*60)//(60*60))])
    n = np.arange((Lw * 7 * 24 * 60 * 60) // (60 * 60))
    return abs(fourier_transform(Xi, f, n))

def FWD(Lw, T, f=1/7):
    Xi =  np.array([studentActivity(24*60*60, T, x_i) for x_i in range((Lw*7*24*60*60)//(24*60*60))])
    n = np.arange((Lw * 7 * 24 * 60 * 60) // (24 * 60 * 60))
    return abs(fourier_transform(Xi, f, n))

def NQZ(Pw, id):
    no_problems_per_student = Pw[['AccountUserID','ProblemID']].groupby('AccountUserID').count()
    return no_problems_per_student.loc[str(id)][0] if str(id) in no_problems_per_student.index else 0.0

def PQZ(Pw, id):
    no_problems_per_year = Pw.groupby(by=["AccountUserID", "Year"]).count()
    if id not in no_problems_per_year.index:
        return 0.0
    temp = no_problems_per_year.loc[str(id)].reset_index().iloc[0]
    return temp[1] / getTotalProblemsFlippedPeriod(temp[0])
