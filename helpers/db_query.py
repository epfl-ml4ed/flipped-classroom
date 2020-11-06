from helpers.db_connector import MySQLConnector
from helpers.data_process import *
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

def getUserInfo(prior_knowledge=False):
    """
    df with columns AccountUserID	Sciper	Round	Condition
    already removed repeating students, students with no grades, and students with only a few video interactions
    """
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'', '\'EPFL-AlgebreLineaire-2019\'']
    columns = ['AccountUserID']
    query = """ SELECT {} FROM ca_courseware.User_Account_Info WHERE DataPackageID in ({}) """.format(", ".join(columns), ", ".join(course_names))
    user_df = queryDB(query, columns) #User info
    video_df = getVideoEvents(isa_only=False) #Video events to filter inactive users
    mapping_df = getMapping() #AccountUserID - Sciper mapping
    condition_df = getStudentCondition(id_anon=True) #Flipped
    user_df = user_df.merge(mapping_df).merge(condition_df)
    user_df = user_df[~user_df['AccountUserID'].isin(getRepeatingStudentsIDs(video_df))] #Remove repeaters
    user_df = user_df[~user_df['AccountUserID'].isin(getUnactiveStudentsIDs(video_df))] #Remove inactives
    user_df = user_df[user_df['SCIPER'].isin(getGrades().StudentSCIPER)] #Remove students without grade
    if prior_knowledge:
        prior_df = getPriorKnowledge()
        user_df = user_df.merge(prior_df).drop(columns=['ID.Anon'])
    return user_df

def getUserLocation():
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'', '\'EPFL-AlgebreLineaire-2019\'']
    columns = ['AccountUserID', 'District', 'Region', 'Country', 'Continent']
    query = """ SELECT {} FROM ca_courseware.User_Location WHERE DataPackageID in ({}) """.format(", ".join(columns), ", ".join(course_names))
    df = queryDB(query, columns)
    return df

def getVideoInfo(mode="base"):
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'', '\'EPFL-AlgebreLineaire-2019\'']
    columns = ['DataPackageID', 'VideoID', 'Title', 'Source']
    columns += [] if mode == 'base' else ['Length', 'OpenTime', 'SoftCloseTime','HardCloseTime']
    query = """ SELECT {} FROM ca_courseware.Video_Info WHERE DataPackageID in ({}) """.format(", ".join(columns), ", ".join(course_names))
    df = queryDB(query, columns)
    return df

def getVideoEventsInfo():
    """
    df with columns AccountUserID	Round	EventType	TimeStamp	Title	Source
    already with only students present in UserInfo
    """
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'','\'EPFL-AlgebreLineaire-2019\'']
    columns = ['AccountUserID', 'EventType', 'TimeStamp', "VideoID", "DataPackageID"]
    query = """ SELECT {} FROM ca_courseware.Video_Events WHERE DataPackageID in ({}) """.format(", ".join(columns), ", ".join(course_names))
    events_df = queryDB(query, columns) # Raw video events
    info_df = getVideoInfo()
    user_df = getUserInfo() # User in the flipped group
    events_df = events_df.merge(user_df).merge(info_df)
    events_df.drop(columns=["DataPackageID", "SCIPER", "VideoID"], inplace=True)
    return events_df

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

def getProblemEventsInfo():
    """
    df with columns AccountUserID	Round	EventType	TimeStamp	ProblemType	MaximumSubmissions
    already with only students present in UserInfo
    """
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'','\'EPFL-AlgebreLineaire-2019\'']
    columns = ['AccountUserID', 'EventType', 'TimeStamp', 'ProblemType', "MaximumSubmissions"]
    query = """ SELECT {} FROM ca_courseware.Problem_Events_with_Info WHERE DataPackageID in ({}) """.format(", ".join(columns), ", ".join(course_names))
    events_df = queryDB(query, columns)
    user_df = getUserInfo()
    
    events_df = events_df.merge(user_df)
    events_df.drop(columns=["SCIPER"], inplace=True)
    return events_df

def getTotalProblemsFlippedPeriod(year):
    course_names = ['\'EPFL-AlgebreLineaire-2017_T3\'', '\'EPFL-AlgebreLineaire-2018\'', '\'EPFL-AlgebreLineaire-2019\'']
    with open('../config/linear_algebra.json') as f:
        config = json.load(f)
    start_flipped = time.mktime(dateutil.parser.parse(config[str(year)]['StartFlipped']).timetuple())
    end_flipped = start_flipped + timedelta(weeks=len(config[str(year)]['FlippedWeeks'])).total_seconds()
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

def getExamInfo():
    """
    df with columns AccountUserID	Round	Grade	GradeDate
    already with only students present in UserInfo 
    """
    exam_df = getFlippedGrades()
    exam_df = exam_df.loc[exam_df.Repeater == 0]
    exam_df.drop(columns=["PlanSection", "PlanCursus", "Repeater", "AcademicYear"], inplace=True)

    user_df = getUserInfo() #Remove repeaters and incative users
    exam_df = exam_df[exam_df['AccountUserID'].isin(user_df['AccountUserID'])]
    return exam_df

def getGrades(flipped = True):
    columns = ['StudentSCIPER', 'AcademicYear', 'Grade', 'GradeDate', 'PlanSection', 'PlanCursus']
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

def getStudentCondition(flipped=True, id_anon=False):
    CONDITION_MAPPING_PATH = '../data/lin_alg_moodle/student-conditions.csv'
    conditions_df = pd.read_csv(CONDITION_MAPPING_PATH, index_col=0)
    # Return either the flipped or the control group
    conditions_df = conditions_df.loc[conditions_df.Condition == ("Flipped" if flipped else "Control")]
    # Remove useless columns and remove duplicates (since a student can take the course during different years)
    conditions_df = conditions_df.drop(columns=["Condition"]).drop_duplicates()
    if not id_anon:
        del conditions_df['ID.Anon']
    conditions_df.rename(columns={"Course.Year":"Round"}, inplace=True)
    return conditions_df

def getFlippedGrades(include_repeaters=True):
    sciper_df = getGrades() # Get the grades by SCIPER
    conditions_df = getStudentCondition() # Get the flipped group list of SCIPER
    # Keep only flipped students 
    sciper_df = sciper_df.merge(conditions_df, left_on='StudentSCIPER', right_on="SCIPER")
    # Get the mapping between Sciper and AccountUserID
    mapping = getMapping()
    # Get the grades by AccountUserID, (1 student dropped here, not in the mapping df)
    userID_df = sciper_df.merge(mapping) 
    # Label the student taking the course for the second time
    userID_df = labelRepeaters(userID_df)
    if not include_repeaters:
        userID_df = userID_df.loc[userID_df.Repeater == 0]
    userID_df.drop(columns=['StudentSCIPER', 'SCIPER'], inplace=True)
    return userID_df

def getControlGrades(include_repeaters=True):
    sciper_df = getGrades(flipped=False) # Get the grades by SCIPER
    conditions_df = getStudentCondition(flipped=False) # Get the Control group list of SCIPER
    # Keep only Control students
    sciper_df = sciper_df.merge(conditions_df, left_on='StudentSCIPER', right_on="SCIPER")
    # Label the student taking the course for the second time
    sciper_df = labelRepeaters(sciper_df)
    if not include_repeaters:
        sciper_df = sciper_df.loc[sciper_df.Repeater == 0]
    sciper_df.drop(columns=['StudentSCIPER', 'SCIPER'], inplace=True)
    return sciper_df

def filterIsaOnly(df):
    #Drop duplicates due to students retaking the class
    isa_id = getFlippedGrades().AccountUserID.drop_duplicates()
    return df.merge(isa_id)

def labelRepeaters(df):
    df["Repeater"] = 0
    #Sort by year so that only second year is labeled repeater
    df.loc[df.sort_values(by="AcademicYear").duplicated("StudentSCIPER"),"Repeater"] = 1
    return df

def getPriorKnowledge(columns = ['ID.Anon', 'Category', "Gender"]):
    FOLDER = "../data/lin_alg_moodle/student_info/"
    PATHS = [FOLDER + "Year{}-Normalized-Score.csv".format(year) for year in range(1,4)]
    year1, year2, year3 = [pd.read_csv(year, index_col=0)[columns] for year in PATHS]
    return pd.concat([year1, year2, year3])