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

def getVideoEvents(mode='base', with_2019 = False, isa_only=False):
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'']
    if with_2019:
        course_names.append('\'EPFL-AlgebreLineaire-2019\'')
    columns = ['AccountUserID', 'DataPackageID', 'VideoID', 'TimeStamp', 'EventType']
    columns +=  [] if mode == 'base' else ['SeekType', 'OldTime', 'CurrentTime', 'NewTime', 'OldSpeed', 'NewSpeed']
    query = """ SELECT {} FROM ca_courseware.Video_Events WHERE DataPackageID in ({}) """.format(", ".join(columns), ", ".join(course_names))
    df = queryDB(query, columns)
    df['Year'] = df['DataPackageID'].apply(lambda x: int(x.split('_')[0][-4:]))
    if isa_only:
        isa_id = getFlippedAccountUserID()
        df = df.merge(isa_id)
    return df

def getProblemEvents(isa_only=False):
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'']
    columns = ['AccountUserID', 'DataPackageID', 'ProblemID', 'TimeStamp', 'EventType', 'ProblemType']
    query = """ SELECT {} FROM ca_courseware.Problem_Events_with_Info WHERE DataPackageID in ({}) """.format(", ".join(columns), ", ".join(course_names))
    df = queryDB(query, columns)
    df['Year'] = df['DataPackageID'].apply(lambda x: int(x.split('_')[0][-4:]))
    if isa_only:
        isa_id = getFlippedAccountUserID()
        df = df.merge(isa_id)
    return df

def getTotalProblemsFlippedPeriod(year):
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'']
    with open('../config/cf_mooc.json') as f:
        config = json.load(f)
    start_flipped = time.mktime(dateutil.parser.parse(config[str(year)]['Start']).timetuple())
    end_flipped = start_flipped + timedelta(weeks=config[str(year)]['FlippedWeeks']).total_seconds()
    query = """SELECT COUNT(DISTINCT ProblemID) FROM ca_courseware.Problem_Events_with_Info WHERE DataPackageID in ({}) AND TimeStamp > {}  AND TimeStamp < {} """.format(", ".join(course_names), start_flipped, end_flipped)
    return queryDB(query, ['NbProblems']).loc[0]['NbProblems']

def getProblemFirstEvents(isa_only=False):
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
        isa_id = getFlippedAccountUserID()
        df = df.merge(isa_id)
    return df

def getTextbookEvents(isa_only=False):
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'', '\'EPFL-AlgebreLineaire-2019\'']
    columns = ['AccountUserID', 'DataPackageID', 'TimeStamp', 'EventType']
    query = """ SELECT {} FROM ca_courseware.TextBook_Events WHERE DataPackageID in ({}) """.format(", ".join(columns), ", ".join(course_names))
    df = queryDB(query, columns)
    df['Year'] = df['DataPackageID'].apply(lambda x: int(x.split('_')[0][-4:]))
    df['Date'] = df.TimeStamp.apply(lambda x: datetime.fromtimestamp(x))
    if isa_only:
        isa_id = getFlippedAccountUserID()
        df = df.merge(isa_id)
    return df

def getForumEvents(isa_only=False):
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'', '\'EPFL-AlgebreLineaire-2019\'']
    columns = ['AccountUserID', 'DataPackageID', 'TimeStamp', 'EventType', 'PostType', 'PostID']
    query = """ SELECT {} FROM ca_courseware.Forum_Events WHERE DataPackageID in ({}) """.format(", ".join(columns), ", ".join(course_names))
    df = queryDB(query, columns)
    df['Year'] = df['DataPackageID'].apply(lambda x: int(x.split('_')[0][-4:]))
    df['Date'] = df.TimeStamp.apply(lambda x: datetime.fromtimestamp(x))
    if isa_only:
        isa_id = getFlippedAccountUserID()
        df = df.merge(isa_id)
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
    userID_df.drop(columns=['StudentSCIPER', 'SCIPER'], inplace=True)    
    return userID_df


def getControlGrades():
    sciper_df = getGrades(flipped=False) # Get the grades by SCIPER
    conditions_df = getStudentCondition(flipped=False) # Get the Control group list of SCIPER
    # Keep only Control students
    sciper_df = sciper_df.merge(conditions_df, left_on='StudentSCIPER', right_on="SCIPER")
    sciper_df.drop(columns=['StudentSCIPER', 'SCIPER'], inplace=True)
    return sciper_df


def getFlippedAccountUserID():
    #Drop duplicates due to students retaking the class
    return getFlippedGrades().AccountUserID.drop_duplicates()