from helpers.db_connector import MySQLConnector
from datetime import datetime, timedelta, date
import dateutil.parser
import pandas as pd
import time
import json

def queryDB(query, labels):
    db = MySQLConnector()
    out = db.execute(query)
    db.close()
    return pd.DataFrame.from_records(out, columns=labels)

def getVideoEvents(mode='base', with_2019 = False):
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'']
    if with_2019:
        course_names.append('\'EPFL-AlgebreLineaire-2019\'')
    columns = ['AccountUserID', 'DataPackageID', 'VideoID', 'TimeStamp', 'EventType']
    columns +=  [] if mode == 'base' else ['SeekType', 'OldTime', 'CurrentTime', 'NewTime', 'OldSpeed', 'NewSpeed']
    query = """ SELECT {} FROM ca_courseware.Video_Events WHERE DataPackageID in ({}) """.format(", ".join(columns), ", ".join(course_names))
    df = queryDB(query, columns)
    df['Year'] = df['DataPackageID'].apply(lambda x: int(x.split('_')[0][-4:]))
    return df

def getProblemEvents():
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'']
    columns = ['AccountUserID', 'DataPackageID', 'ProblemID', 'TimeStamp', 'EventType', 'ProblemType']
    query = """ SELECT {} FROM ca_courseware.Problem_Events_with_Info WHERE DataPackageID in ({}) """.format(", ".join(columns), ", ".join(course_names))
    df = queryDB(query, columns)
    df['Year'] = df['DataPackageID'].apply(lambda x: int(x.split('_')[0][-4:]))
    return df

def getTotalProblemsFlippedPeriod(year):
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'']
    with open('../config/cf_mooc.json') as f:
        config = json.load(f)
    start_flipped = time.mktime(dateutil.parser.parse(config[str(year)]['Start']).timetuple())
    end_flipped = start_flipped + timedelta(weeks=config[str(year)]['FlippedWeeks']).total_seconds()
    query = """SELECT COUNT(DISTINCT ProblemID) FROM ca_courseware.Problem_Events_with_Info WHERE DataPackageID in ({}) AND TimeStamp > {}  AND TimeStamp < {} """.format(", ".join(course_names), start_flipped, end_flipped)
    return queryDB(query, ['NbProblems']).loc[0]['NbProblems']

def getProblemFirstEvents():
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'']
    columns = ['AccountUserID', 'DataPackageID', 'ProblemID', 'TimeStamp', 'EventType', 'ProblemType']
    query = """SELECT {columns}
    FROM (
        SELECT {columns}, ROW_NUMBER() OVER(PARTITION BY AccountUserID, ProblemID 
                                            ORDER BY AccountUserID, ProblemID, TimeStamp) rn
        FROM ca_courseware.Problem_Events_with_Info
        WHERE DataPackageID in ({courses})) t
    WHERE rn = 1;
    """.format(columns=", ".join(columns), courses=", ".join(course_names))
    df = queryDB(query, columns)
    df['Year'] = df['DataPackageID'].apply(lambda x: int(x.split('_')[0][-4:]))
    return df

def getTextbookEvents():
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'', '\'EPFL-AlgebreLineaire-2019\'']
    columns = ['AccountUserID', 'DataPackageID', 'TimeStamp', 'EventType']
    query = """ SELECT {} FROM ca_courseware.TextBook_Events WHERE DataPackageID in ({}) """.format(", ".join(columns), ", ".join(course_names))
    df = queryDB(query, columns)
    df['Year'] = df['DataPackageID'].apply(lambda x: int(x.split('_')[0][-4:]))
    df['Date'] = df.TimeStamp.apply(lambda x: datetime.fromtimestamp(x))
    return df

def getForumEvents():
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'', '\'EPFL-AlgebreLineaire-2019\'']
    columns = ['AccountUserID', 'DataPackageID', 'TimeStamp', 'EventType', 'PostType', 'PostID']
    query = """ SELECT {} FROM ca_courseware.Forum_Events WHERE DataPackageID in ({}) """.format(", ".join(columns), ", ".join(course_names))
    df = queryDB(query, columns)
    df['Year'] = df['DataPackageID'].apply(lambda x: int(x.split('_')[0][-4:]))
    df['Date'] = df.TimeStamp.apply(lambda x: datetime.fromtimestamp(x))
    return df