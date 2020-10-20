from helpers.db_connector import MySQLConnector
from datetime import timedelta, datetime
import dateutil.parser
import pandas as pd
import time
import json

def queryDB(query, labels):
    db = MySQLConnector()
    out = db.execute(query)
    db.close()
    return pd.DataFrame.from_records(out, columns=labels)

def getUserDemo():
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'', '\'EPFL-AlgebreLineaire-2019\'']
    columns = ['AccountUserID', 'Gender', 'Age', 'HighestDegree']
    query = """ SELECT {} FROM ca_courseware.User_Demographics WHERE DataPackageID in ({}) """.format(", ".join(columns), ", ".join(course_names))
    df = queryDB(query, columns)
    df['Year'] = df['DataPackageID'].apply(lambda x: int(x.split('_')[0][-4:]))
    return df

def getUserInfo():
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'', '\'EPFL-AlgebreLineaire-2019\'']
    columns = ['AccountUserID', 'DataPackageID', 'Timestamp', 'ISOWeekDate']
    query = """ SELECT {} FROM ca_courseware.User_Account_Info WHERE DataPackageID in ({}) """.format(", ".join(columns), ", ".join(course_names))
    df = queryDB(query, columns)
    df['Year'] = df['DataPackageID'].apply(lambda x: int(x.split('_')[0][-4:]))
    df['Week'] = df['ISOWeekDate'].apply(lambda x: x.split('-')[1])
    df['Weekday'] = df['ISOWeekDate'].apply(lambda x: x.split('-')[2])
    del df['DataPackageID']
    del df['ISOWeekDate']
    return df

def getUserLocation():
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'', '\'EPFL-AlgebreLineaire-2019\'']
    columns = ['AccountUserID', 'District', 'Region', 'Country', 'Continent']
    query = """ SELECT {} FROM ca_courseware.User_Location WHERE DataPackageID in ({}) """.format(", ".join(columns), ", ".join(course_names))
    df = queryDB(query, columns)
    return df

def getVideoInfo():
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'', '\'EPFL-AlgebreLineaire-2019\'']
    columns = ['DataPackageID', 'VideoID', 'Title', 'Source']
    query = """ SELECT {} FROM ca_courseware.Video_Info WHERE DataPackageID in ({}) """.format(", ".join(columns), ", ".join(course_names))
    df = queryDB(query, columns)
    return df

def getVideoEvents(mode='base'):
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'', '\'EPFL-AlgebreLineaire-2019\'']
    columns = ['DataPackageID', 'AccountUserID', 'VideoID', 'TimeStamp', 'EventType']
    columns +=  [] if mode == 'base' else ['SeekType', 'OldTime', 'CurrentTime', 'NewTime', 'OldSpeed', 'NewSpeed']
    query = """ SELECT {} FROM ca_courseware.Video_Events WHERE DataPackageID in ({}) """.format(", ".join(columns), ", ".join(course_names))
    df = queryDB(query, columns)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S')))
    df['Year'] = df['DataPackageID'].apply(lambda x: int(x.split('_')[0][-4:]))
    return df

def getTextbookEvents():
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'', '\'EPFL-AlgebreLineaire-2019\'']
    columns = ['AccountUserID', 'DataPackageID', 'TimeStamp', 'EventType', 'OldPage', 'NewPage', 'CurrentZoom', 'ScrollDirection']
    query = """ SELECT {} FROM ca_courseware.TextBook_Events WHERE DataPackageID in ({}) """.format(", ".join(columns), ", ".join(course_names))
    df = queryDB(query, columns)
    df['Year'] = df['DataPackageID'].apply(lambda x: int(x.split('_')[0][-4:]))
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S')))
    del df['DataPackageID']
    return df

def getProblemInfo():
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'', '\'EPFL-AlgebreLineaire-2019\'']
    columns = ['DataPackageID', 'ProblemID', 'ProblemType', 'MaximumSubmissions', 'Title']
    query = """ SELECT {} FROM ca_courseware.Problem_Info WHERE DataPackageID in ({}) """.format(", ".join(columns), ", ".join(course_names))
    df = queryDB(query, columns)
    df['Year'] = df['DataPackageID'].apply(lambda x: int(x.split('_')[0][-4:]))
    return df

def getProblemEvents():
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'', '\'EPFL-AlgebreLineaire-2019\'']
    columns = ['DataPackageID', 'AccountUserID', 'ProblemID', 'TimeStamp', 'EventType']
    query = """ SELECT {} FROM ca_courseware.Problem_Events_with_Info WHERE DataPackageID in ({}) """.format(", ".join(columns), ", ".join(course_names))
    df = queryDB(query, columns)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S')))
    return df

def getForumInfo():
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'', '\'EPFL-AlgebreLineaire-2019\'']
    columns = ['DataPackageID', 'PostID', 'PostType', 'ParentPostType', 'ParentPostID', 'PostTitle', 'PostText']
    query = """ SELECT {} FROM ca_courseware.Forum_Info WHERE DataPackageID in ({}) """.format(", ".join(columns), ", ".join(course_names))
    df = queryDB(query, columns)
    df['Year'] = df['DataPackageID'].apply(lambda x: int(x.split('_')[0][-4:]))
    return df

def getForumEvents():
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'', '\'EPFL-AlgebreLineaire-2019\'']
    columns = ['DataPackageID', 'AccountUserID', 'TimeStamp', 'EventType', 'PostID']
    query = """ SELECT {} FROM ca_courseware.Forum_Events WHERE DataPackageID in ({}) """.format(", ".join(columns), ", ".join(course_names))
    df = queryDB(query, columns)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S')))
    return df

def getTotalProblemsFlippedPeriod(year):
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'', '\'EPFL-AlgebreLineaire-2019\'']
    with open('../config/cf_mooc.json') as f:
        config = json.load(f)
    start_flipped = time.mktime(dateutil.parser.parse(config[str(year)]['Start']).timetuple())
    end_flipped = start_flipped + timedelta(weeks=config[str(year)]['FlippedWeeks']).total_seconds()
    query = """SELECT COUNT(DISTINCT ProblemID) FROM ca_courseware.Problem_Events_with_Info WHERE DataPackageID in ({}) AND TimeStamp > {}  AND TimeStamp < {} """.format(", ".join(course_names), start_flipped, end_flipped)
    return queryDB(query, ['NbProblems']).loc[0]['NbProblems']