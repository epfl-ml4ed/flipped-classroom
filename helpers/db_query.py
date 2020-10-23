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

def getUserDemo():
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'', '\'EPFL-AlgebreLineaire-2019\'']
    columns = ['AccountUserID', 'Gender', 'Age', 'HighestDegree']
    query = """ SELECT {} FROM ca_courseware.User_Demographics WHERE DataPackageID in ({}) """.format(", ".join(columns), ", ".join(course_names))
    df = queryDB(query, columns)
    df['Year'] = df['DataPackageID'].apply(lambda x: int(x.split('_')[0][-4:]))
    return df

def getUserInfo(isa_only=True):
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'', '\'EPFL-AlgebreLineaire-2019\'']
    columns = ['AccountUserID', 'DataPackageID', 'Timestamp', 'ISOWeekDate']
    query = """ SELECT {} FROM ca_courseware.User_Account_Info WHERE DataPackageID in ({}) """.format(", ".join(columns), ", ".join(course_names))
    df = queryDB(query, columns)
    df['Year'] = df['DataPackageID'].apply(lambda x: int(x.split('_')[0][-4:]))
    df['Week'] = df['ISOWeekDate'].apply(lambda x: x.split('-')[1])
    df['Weekday'] = df['ISOWeekDate'].apply(lambda x: x.split('-')[2])
    del df['DataPackageID']
    del df['ISOWeekDate']
    if isa_only:
        df = filterIsaOnly(df)
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

def getVideoEvents(mode='base', isa_only=True):
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'','\'EPFL-AlgebreLineaire-2019\'']
    columns = [ 'DataPackageID', 'AccountUserID', 'VideoID', 'TimeStamp', 'EventType']
    columns +=  [] if mode == 'base' else ['SeekType', 'OldTime', 'CurrentTime', 'NewTime', 'OldSpeed', 'NewSpeed']
    query = """ SELECT {} FROM ca_courseware.Video_Events WHERE DataPackageID in ({}) """.format(", ".join(columns), ", ".join(course_names))
    df = queryDB(query, columns)
    df['Date'] = pd.to_datetime(df['TimeStamp'].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S')))
    df['Year'] = df['DataPackageID'].apply(lambda x: int(x.split('_')[0][-4:]))
    if isa_only:
        df = filterIsaOnly(df)
    return df

def getTextbookEvents(isa_only=True):
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'', '\'EPFL-AlgebreLineaire-2019\'']
    columns = ['AccountUserID', 'DataPackageID', 'TimeStamp', 'EventType', 'OldPage', 'NewPage', 'CurrentZoom', 'ScrollDirection']
    query = """ SELECT {} FROM ca_courseware.TextBook_Events WHERE DataPackageID in ({}) """.format(", ".join(columns), ", ".join(course_names))
    df = queryDB(query, columns)
    df['Year'] = df['DataPackageID'].apply(lambda x: int(x.split('_')[0][-4:]))
    df['Date'] = pd.to_datetime(df['TimeStamp'].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S')))
    del df['DataPackageID']
    if isa_only:
        df = filterIsaOnly(df)
    return df

def getProblemInfo():
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'', '\'EPFL-AlgebreLineaire-2019\'']
    columns = ['DataPackageID', 'ProblemID', 'ProblemType', 'MaximumSubmissions', 'Title']
    query = """ SELECT {} FROM ca_courseware.Problem_Info WHERE DataPackageID in ({}) """.format(", ".join(columns), ", ".join(course_names))
    df = queryDB(query, columns)
    df['Year'] = df['DataPackageID'].apply(lambda x: int(x.split('_')[0][-4:]))
    return df

def getProblemEvents(isa_only=True):
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'','\'EPFL-AlgebreLineaire-2019\'']
    columns = ['DataPackageID', 'AccountUserID', 'ProblemID', 'TimeStamp', 'EventType']
    query = """ SELECT {} FROM ca_courseware.Problem_Events_with_Info WHERE DataPackageID in ({}) """.format(", ".join(columns), ", ".join(course_names))
    df = queryDB(query, columns)
    df['Date'] = pd.to_datetime(df['TimeStamp'].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S')))
    if isa_only:
        df = filterIsaOnly(df)
    return df

def getForumInfo():
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'', '\'EPFL-AlgebreLineaire-2019\'']
    columns = ['DataPackageID', 'PostID', 'PostType', 'ParentPostType', 'ParentPostID', 'PostTitle', 'PostText']
    query = """ SELECT {} FROM ca_courseware.Forum_Info WHERE DataPackageID in ({}) """.format(", ".join(columns), ", ".join(course_names))
    df = queryDB(query, columns)
    df['Year'] = df['DataPackageID'].apply(lambda x: int(x.split('_')[0][-4:]))
    return df

def getForumEvents(isa_only=True):
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'', '\'EPFL-AlgebreLineaire-2019\'']
    columns = ['DataPackageID', 'AccountUserID', 'TimeStamp', 'EventType', 'PostID']
    query = """ SELECT {} FROM ca_courseware.Forum_Events WHERE DataPackageID in ({}) """.format(", ".join(columns), ", ".join(course_names))
    df = queryDB(query, columns)
    df['Year'] = df['DataPackageID'].apply(lambda x: int(x.split('_')[0][-4:]))
    df['Date'] = pd.to_datetime(df['TimeStamp'].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S')))
    if isa_only:
        df = filterIsaOnly(df)
    return df

def getTotalProblemsFlippedPeriod(year):
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'', '\'EPFL-AlgebreLineaire-2019\'']
    with open('../config/cf_mooc.json') as f:
        config = json.load(f)
    start_flipped = time.mktime(dateutil.parser.parse(config[str(year)]['Start']).timetuple())
    end_flipped = start_flipped + timedelta(weeks=config[str(year)]['FlippedWeeks']).total_seconds()
    query = """SELECT COUNT(DISTINCT ProblemID) FROM ca_courseware.Problem_Events_with_Info WHERE DataPackageID in ({}) AND TimeStamp > {}  AND TimeStamp < {} """.format(", ".join(course_names), start_flipped, end_flipped)
    return queryDB(query, ['NbProblems']).loc[0]['NbProblems']

def getProblemFirstEvents(isa_only=True):
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
    if isa_only:
        df = filterIsaOnly(df)
    return df


def getGrades(flipped = True):
    columns = ['StudentSCIPER', 'AcademicYear', 'Grade', 'PlanSection', 'PlanCursus']
    query = """ SELECT distinct {} FROM project_himanshu.Bachelor_Master_Results
            """.format(", ".join(columns))
    if flipped:
        query += "WHERE TeacherSCIPER = 121157 AND SubjectName = 'Algèbre linéaire (classe inversée)'"
    else:
        query += "WHERE SubjectName = 'Algèbre linéaire'"
    sciper_df = queryDB(query, columns)
    sciper_df.Grade = pd.to_numeric(sciper_df.Grade, errors="coerce") # Convert grades to Float
    sciper_df = sciper_df[~pd.isna(sciper_df).any(axis=1)] # Drop NaN
    return sciper_df

def getMapping():
    columns = ['AccountUserID', 'SCIPER']
    query = """ SELECT {} FROM project_himanshu.MOOC_ISA_Person_Mapping""".format(", ".join(columns))
    mapping = queryDB(query, columns)
    return mapping

def getStudentCondition(flipped=True):
    CONDITION_MAPPING_PATH = '../data/lin_alg_moodle/Volunteer-Flipped-Proj.csv'
    conditions_df = pd.read_csv(CONDITION_MAPPING_PATH, index_col=0)
    # Return either the flipped or the control group
    conditions_df = conditions_df.loc[conditions_df.Condition == ("Flipped" if flipped else "Control")]
    # Remove useless columns and remove duplicates (since a student can take the course during different years)
    conditions_df = conditions_df.drop(columns=["Course.Year", "Condition"]).drop_duplicates()
    return conditions_df
    
        
def getFlippedGrades():
    sciper_df = getGrades() # Get the grades by SCIPER
    conditions_df = getStudentCondition() # Get the flipped group list of SCIPER
    # Keep only flipped students 
    sciper_df = sciper_df.merge(conditions_df, left_on='StudentSCIPER', right_on="SCIPER")
    # Get the mapping between Sciper and AccountUserID
    mapping = getMapping()
    # Get the grades by AccountUserID, (1 student dropped here, not in the mapping df)
    userID_df = sciper_df.merge(mapping) 
    # Label the student taking the course for the second time
    userID_df = labelRepentants(userID_df)
    userID_df.drop(columns=['StudentSCIPER', 'SCIPER'], inplace=True)    
    return userID_df


def getControlGrades():
    sciper_df = getGrades(flipped=False) # Get the grades by SCIPER
    conditions_df = getStudentCondition(flipped=False) # Get the Control group list of SCIPER
    # Keep only Control students
    sciper_df = sciper_df.merge(conditions_df, left_on='StudentSCIPER', right_on="SCIPER")
    # Label the student taking the course for the second time
    sciper_df = labelRepentants(sciper_df)
    sciper_df.drop(columns=['StudentSCIPER', 'SCIPER'], inplace=True)
    return sciper_df

def filterIsaOnly(df):
    #Drop duplicates due to students retaking the class
    isa_id = getFlippedGrades().AccountUserID.drop_duplicates()
    return df.merge(isa_id)

def labelRepentants(df):
    df["Repentant"] = 0
    #Sort by year so that only second year is labeled repentant
    df.loc[df.sort_values(by="AcademicYear").duplicated("StudentSCIPER"),"Repentant"] = 1
    return df