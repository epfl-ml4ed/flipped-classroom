from datetime import datetime

def string_to_datetime(ts):
    return datetime.strptime(ts, '%Y-%m-%d')

def datetime_to_week(dt):
    return dt.strftime('%W')

def drop_date_outsiders(series, start_date, end_date):
    start_dt = string_to_datetime(start_date)
    end_dt = string_to_datetime(end_date)
    return series[(series >= start_dt) & (series <= end_dt)]

def order_week(series, start_date):
    start_week = int(datetime_to_week(string_to_datetime(start_date)))
    nb_week = int(series.max())
    return series.apply(lambda w: str((int(w) - start_week) % nb_week))

def processYear(series, start_date, end_date):
    series = drop_date_outsiders(series, start_date, end_date)
    series = series.apply(datetime_to_week)
    return order_week(series,start_date)