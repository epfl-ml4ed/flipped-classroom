from datetime import datetime

def getSeconds(row):
    if 'Start' in row and 'End' in row:
        return (row['End'] - row['Start']).total_seconds()
    return 0.0

def string2Datetime(ts):
    return datetime.strptime(ts, '%Y-%m-%d')

def datetime2Week(dt):
    return dt.strftime('%W')

def dropDateOutsiders(series, start_date, end_date):
    start_dt = string2Datetime(start_date)
    end_dt = string2Datetime(end_date)
    return series[(series >= start_dt) & (series <= end_dt)]

def orderWeek(series, start_date):
    start_week = int(datetime2Week(string2Datetime(start_date)))
    nb_week = int(series.max())
    return series.apply(lambda w: str((int(w) - start_week) % nb_week))

def processYear(series, start_date, end_date):
    series = dropDateOutsiders(series, start_date, end_date)
    series = series.apply(datetime2Week)
    return orderWeek(series,start_date)

def getCourseWeek(series, start_date, end_date):
    series = dropDateOutsiders(series, start_date, end_date)
    series = series.apply(datetime2Week)
    return orderWeek(series,start_date)

