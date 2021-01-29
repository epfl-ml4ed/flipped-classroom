#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from datetime import datetime

import pandas as pd

from helper.database.db_connector import MySQLConnector


def str2dt(ts):
    return datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')

def tmp2dt(x):
    return str2dt(datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))

def downloadLabelsFromISA(config, metadata):
    db = MySQLConnector()

    for course in config['courses']:
        print('> retrieving labels for', course, '...')
        title = metadata[metadata['DataPackageID'] == '_'.join(course.split('.')[0].split('_')[2:])]['OfficialTitle'].values[0]

        labels = pd.DataFrame.from_records(db.execute("""SELECT StudentPersonHash, Grade, GradeDate FROM project_himanshu.Request_2021Jan26_Bachelor_Master_Results WHERE SubjectName LIKE \'%{}%\'""".format(title)), columns=['StudentSCIPER', 'Grade', 'GradeDate']).drop_duplicates()
        labels = labels[~labels['Grade'].isin(['STATUT_NOTE_NA', 'STATUT_NOTE_M'])]
        print('> found labels for', len(labels), 'students in the course across years')

        data_package = course.split('.')[0].split('_')[2:][0]
        videosEventsNew = pd.DataFrame.from_records(db.execute("""SELECT AccountUserHash, TimeStamp FROM project_himanshu.Request_2021Jan26_Video_Events WHERE DataPackageID LIKE \'%{}%\' ORDER BY TimeStamp""".format(data_package)), columns=['AccountUserID', 'TimeStamp']).drop_duplicates(subset=['AccountUserID'], keep='first')
        videosEventsOld = pd.DataFrame.from_records(db.execute("""SELECT AccountUserID, TimeStamp FROM ca_courseware.Video_Events WHERE DataPackageID LIKE \'%{}%\' ORDER BY TimeStamp""".format(data_package)), columns=['OldAccountUserID', 'TimeStamp']).drop_duplicates(subset=['OldAccountUserID'], keep='first')
        videosEvents = videosEventsNew.merge(videosEventsOld, on='TimeStamp')
        oldMapping = {i:v for i, v in zip(videosEvents['AccountUserID'].values, videosEvents['OldAccountUserID'].values)}
        print('> found', len(videosEvents['AccountUserID'].unique()), 'students with video events in this year course')

        mapping = pd.DataFrame.from_records(db.execute("""SELECT AccountUserHash, StudentPersonHash FROM project_himanshu.Request_2021Jan26_ISA_MOOCs_Mapping"""), columns=['AccountUserID', 'StudentSCIPER'])
        course_mapping = mapping[mapping['AccountUserID'].isin(videosEvents['AccountUserID'].astype(str))].copy()
        course_mapping['OldAccountUserID'] = course_mapping['AccountUserID'].apply(lambda x: oldMapping[x])
        print('> found a mooc-isa map for', len(course_mapping), 'students in this year course')

        courseLabels = course_mapping.merge(labels, how='left', on='StudentSCIPER').dropna(subset=['Grade'])[['AccountUserID', 'OldAccountUserID', 'Grade', 'GradeDate']]
        courseLabels['GradeMax'] = 6.0
        print('> retrieved labels for', len(courseLabels), 'students in this year course')
        print('> found counts of labels as', courseLabels['Grade'].value_counts().sort_index().to_dict())

        courseLabels[['AccountUserID', 'OldAccountUserID', 'Grade', 'GradeMax', 'GradeDate']].to_csv(os.path.join('request_2021jan26_labels', 'request_2021jan26_final_grades_' + course + '.csv'), index=False)

        print()

    db.close()

def downloadQuizLabelsFromCoursera(config, base_folder='./problem_events'):
    db = MySQLConnector()

    for cindex, course in enumerate(os.listdir(base_folder)):
        if cindex == 3:
            break
        if 'progfun' in course or not 'coursera' in course:
            continue

        platform = '_'.join(course.split('.')[0].split('_')[2:4])
        package = '_'.join(course.split('.')[0].split('_')[4:])

        problems = pd.DataFrame.from_records(db.execute("""SELECT DISTINCT ProblemID FROM {}.Problem_Info WHERE DataPackageID = \'{}\' AND ProblemType = 'Quiz' AND Title IS NOT NULL and (Title NOT LIKE '%Sample%' AND Title NOT LIKE '%Survey%' AND Title NOT LIKE '%New%' AND Title NOT LIKE '%Tutorial%' AND Title NOT LIKE '%Mastery%' AND Title NOT LIKE '%Basic%' AND Title NOT LIKE '%Advanced%' AND Title NOT LIKE 'exercice1' AND Title NOT LIKE 'Exercice 1' AND Title NOT LIKE '%Devoirs%' AND Title NOT LIKE '%préliminaire%') AND MaximumSubmissions >= 1""".format(platform, package)), columns=['ProblemID'])['ProblemID'].tolist()
        problemsEvents = pd.DataFrame.from_records(db.execute("""SELECT SessionUserID, ProblemID, ProblemType, Title, Grade, SubmissionNumber, TimeStamp FROM {}.Problem_Events_with_Info WHERE DataPackageID = \'{}\' AND Grade IS NOT NULL""".format(platform, package)), columns=['AccountUserID', 'ProblemID', 'ProblemType', 'ProblemTitle', 'Grade', 'SubmissionNumber', 'GradeDate']).drop_duplicates()

        if len(problems) > 0:
            ngrades = problemsEvents.groupby(by='ProblemTitle')['Grade'].max()

            problems = list(set(problems) & set(problemsEvents['ProblemID'].unique()))
            problemsEvents = problemsEvents[problemsEvents['ProblemID'].isin(problems)].sort_values(by=['AccountUserID', 'ProblemID', 'SubmissionNumber']).drop_duplicates(subset=['AccountUserID', 'ProblemID'], keep=config['keep'])

            nproblems = problemsEvents.groupby(by='AccountUserID').size()
            maxproblems = max(nproblems.values)
            filtered_user_ids = nproblems[nproblems == maxproblems].index.to_list()
            problemsEvents = problemsEvents[problemsEvents['AccountUserID'].isin(filtered_user_ids)].sort_values(by='GradeDate').reset_index(drop=True)
            problemsEvents['ProblemMax'] = maxproblems

            maxngrades = {k:v for k,v in zip(ngrades.index, ngrades.values)}
            problemsEvents['GradeMax'] = problemsEvents['ProblemTitle'].apply(lambda x: maxngrades[x])
            problemsEvents['GradeDate'] = problemsEvents['GradeDate'].apply(tmp2dt)

            y = problemsEvents[['AccountUserID', 'ProblemID', 'ProblemType', 'ProblemTitle', 'ProblemMax', 'Grade', 'GradeMax', 'GradeDate']]

            y.to_csv(os.path.join('labels', 'labels_' + course), index=False)
            print(course, len(y['AccountUserID'].unique()))
        else:
            print('> skipped', course)

    db.close()

def downloadAssignmentsLabelsFromCoursera(config, base_folder='./problem_events'):
    db = MySQLConnector()

    for course in os.listdir(base_folder):
        if (not 'structures' in course) or not 'coursera' in course:
            continue

        platform = '_'.join(course.split('.')[0].split('_')[2:4])
        package = '_'.join(course.split('.')[0].split('_')[4:])

        problems = pd.DataFrame.from_records(db.execute("""SELECT DISTINCT Title FROM {}.Problem_Info WHERE DataPackageID = \'{}\' AND ProblemType = 'Assignment' and Title IS NOT NULL AND Title NOT LIKE '%Example%' AND Title NOT LIKE '%Sample%'and Title NOT LIKE '%Bloxorz%'""".format(platform, package)), columns=['ProblemID'])['ProblemID'].tolist()
        problemsEvents = pd.DataFrame.from_records(db.execute("""SELECT SessionUserID, ProblemID, ProblemType, Title, Grade, SubmissionNumber, TimeStamp FROM {}.Problem_Events_with_Info WHERE DataPackageID = \'{}\' AND Grade IS NOT NULL""".format(platform, package)), columns=['AccountUserID', 'ProblemID', 'ProblemType', 'ProblemTitle', 'Grade', 'SubmissionNumber', 'GradeDate']).drop_duplicates()

        print(course)
        if not 'structures' in course:
            problems = [p + ' / ' + p for p in problems]
        elif not 'structures-' in course:
            problems = ['Résultats / Exercice 25','Résultats / Exercice 21', 'Résultats / Exercice 22', 'Résultats / Exercice 24', 'Résultats / Exercice 28', 'Résultats / Exercice 23']
        else:
            problems = ['Résultats / Exercice 1']

        if len(problems) > 0:

            problems = list(set(problems) & set(problemsEvents['ProblemTitle'].unique()))
            problemsEvents = problemsEvents[problemsEvents['ProblemTitle'].isin(problems)].sort_values(by=['AccountUserID', 'ProblemID', 'SubmissionNumber']).drop_duplicates(subset=['AccountUserID', 'ProblemID'], keep=config['keep'])

            ngrades = problemsEvents.groupby(by='ProblemTitle')['Grade'].max()

            nproblems = problemsEvents.groupby(by='AccountUserID').size()
            maxproblems = max(nproblems.values)

            filtered_user_ids = nproblems[nproblems == maxproblems].index.to_list()
            problemsEvents = problemsEvents[problemsEvents['AccountUserID'].isin(filtered_user_ids)].sort_values(by='GradeDate').reset_index(drop=True)
            problemsEvents['ProblemMax'] = maxproblems

            maxngrades = {k:v for k,v in zip(ngrades.index, ngrades.values)}
            problemsEvents['GradeMax'] = problemsEvents['ProblemTitle'].apply(lambda x: maxngrades[x]) if not 'structures' in course else 100.00
            problemsEvents['GradeDate'] = problemsEvents['GradeDate'].apply(tmp2dt)

            y = problemsEvents[['AccountUserID', 'ProblemID', 'ProblemType', 'ProblemTitle', 'ProblemMax', 'Grade', 'GradeMax', 'GradeDate']].sort_values(by=['AccountUserID', 'GradeDate'])

            if len(y['AccountUserID'].unique()) > 10:
                y.to_csv(os.path.join('labels', 'labels_' + course), index=False)
            print(course, len(problems), len(y['AccountUserID'].unique()))
        else:
            print('> skipped', course)

    db.close()

def downloadAssignmentsLabelsFromEdx(config, base_folder='./problem_events'):
    db = MySQLConnector()

    for course in os.listdir(base_folder):
        if not 'edx' in course:
            continue

        platform = '_'.join(course.split('.')[0].split('_')[2:4])
        package = '_'.join(course.split('.')[0].split('_')[4:])

        problems = pd.DataFrame.from_records(db.execute("""SELECT DISTINCT Title FROM {}.Problem_Info WHERE ProblemType = 'Assignment' AND DataPackageID = \'{}\' AND MaximumSubmissions IS NOT NULL AND MaximumSubmissions >= 1""".format(platform, package)), columns=['ProblemID'])['ProblemID'].tolist()
        problemsEvents = pd.DataFrame.from_records(db.execute("""SELECT AccountUserID, ProblemID, ProblemType, Title, Grade, SubmissionNumber, TimeStamp FROM {}.Problem_Events_with_Info WHERE DataPackageID = \'{}\' AND Grade IS NOT NULL""".format(platform, package)), columns=['AccountUserID', 'ProblemID', 'ProblemType', 'ProblemTitle', 'Grade', 'SubmissionNumber', 'GradeDate']).drop_duplicates()

        if len(problems) > 2 and None not in problems:
            ngrades = problemsEvents.groupby(by='ProblemTitle')['Grade'].max()

            problems = list(set(problems) & set(problemsEvents['ProblemTitle'].unique()))
            problemsEvents = problemsEvents[problemsEvents['ProblemTitle'].isin(problems)].sort_values(by=['AccountUserID', 'ProblemID', 'SubmissionNumber']).drop_duplicates(subset=['AccountUserID', 'ProblemID'], keep=config['keep'])

            if len(problemsEvents) > 0:
                nproblems = problemsEvents.groupby(by='AccountUserID').size()
                maxproblems = max(nproblems.values)
                filtered_user_ids = nproblems[nproblems == maxproblems].index.to_list()
                problemsEvents = problemsEvents[problemsEvents['AccountUserID'].isin(filtered_user_ids)].sort_values(by='GradeDate').reset_index(drop=True)
                problemsEvents['ProblemMax'] = maxproblems

                maxngrades = {k:v for k,v in zip(ngrades.index, ngrades.values)}
                problemsEvents['GradeMax'] = problemsEvents['ProblemTitle'].apply(lambda x: maxngrades[x])
                problemsEvents['GradeDate'] = problemsEvents['GradeDate'].apply(tmp2dt)

                y = problemsEvents[['AccountUserID', 'ProblemID', 'ProblemType', 'ProblemTitle', 'ProblemMax', 'Grade', 'GradeMax', 'GradeDate']]

                if 'Algebre' in course or 'EE585' in course or 'FndBioImgx' in course or 'IPA4LSx-3T' in course or 'memsX-3T' in course or 'NMR101x' in course or 'PHYS-209' in course:
                    y.to_csv(os.path.join('labels', 'labels_' + course), index=False)
                print(course, len(problems), maxproblems, len(y['AccountUserID'].unique()))
            else:
                print('> skipped1', course)
        else:
            print('> skipped2', course)

    db.close()

def downloadQuizLabelsFromEdx(config, base_folder='./problem_events'):
    db = MySQLConnector()

    for course in os.listdir(base_folder):
        if not 'edx' in course:
            continue

        platform = '_'.join(course.split('.')[0].split('_')[2:4])
        package = '_'.join(course.split('.')[0].split('_')[4:])

        problems = pd.DataFrame.from_records(db.execute("""SELECT DISTINCT Title FROM {}.Problem_Info WHERE ProblemType IN ('Quiz', 'Assignment') AND DataPackageID = \'{}\' AND MaximumSubmissions IS NOT NULL AND MaximumSubmissions >= 1""".format(platform, package)), columns=['ProblemID'])['ProblemID'].tolist()
        problemsEvents = pd.DataFrame.from_records(db.execute("""SELECT AccountUserID, ProblemID, ProblemType, Title, Grade, SubmissionNumber, TimeStamp FROM {}.Problem_Events_with_Info WHERE DataPackageID = \'{}\' AND Grade IS NOT NULL""".format(platform, package)), columns=['AccountUserID', 'ProblemID', 'ProblemType', 'ProblemTitle', 'Grade', 'SubmissionNumber', 'GradeDate']).drop_duplicates()

        if len(problems) > 2 and None not in problems:
            ngrades = problemsEvents.groupby(by='ProblemTitle')['Grade'].max()

            problems = list(set(problems) & set(problemsEvents['ProblemTitle'].unique()))
            problemsEvents = problemsEvents[problemsEvents['ProblemTitle'].isin(problems)].sort_values(by=['AccountUserID', 'ProblemID', 'SubmissionNumber']).drop_duplicates(subset=['AccountUserID', 'ProblemID'], keep=config['keep'])

            if len(problemsEvents) > 0:
                nproblems = problemsEvents.groupby(by='AccountUserID').size()
                maxproblems = max(nproblems.values)
                filtered_user_ids = nproblems[nproblems == maxproblems].index.to_list()
                problemsEvents = problemsEvents[problemsEvents['AccountUserID'].isin(filtered_user_ids)].sort_values(by='GradeDate').reset_index(drop=True)
                problemsEvents['ProblemMax'] = maxproblems

                maxngrades = {k:v for k,v in zip(ngrades.index, ngrades.values)}
                problemsEvents['GradeMax'] = problemsEvents['ProblemTitle'].apply(lambda x: maxngrades[x])
                problemsEvents['GradeDate'] = problemsEvents['GradeDate'].apply(tmp2dt)

                y = problemsEvents[['AccountUserID', 'ProblemID', 'ProblemType', 'ProblemTitle', 'ProblemMax', 'Grade', 'GradeMax', 'GradeDate']]

                if 'Brain' in course or 'CS305' in course or 'EE-100B' in course or 'SmartCitiesX' in course or 'SynchrotronsX' in course or 'UrbanInfrastructuresX' in course:
                    y.to_csv(os.path.join('labels', 'labels_' + course), index=False)
                print(course, len(problems), maxproblems, len(y['AccountUserID'].unique()))
            else:
                print('> skipped1', course)
        else:
            print('> skipped2', course)

    db.close()

def downloadAssignmentsLabelsFromCourseware(config, base_folder='./problem_events'):
    db = MySQLConnector()

    for course in os.listdir(base_folder):
        if not 'courseware' in course:
            continue

        platform = '_'.join(course.split('.')[0].split('_')[2:4])
        package = '_'.join(course.split('.')[0].split('_')[4:])

        problems = pd.DataFrame.from_records(db.execute("""SELECT DISTINCT ProblemID FROM {}.Problem_Info WHERE ProblemType = 'Quiz' AND DataPackageID = \'{}\' AND MaximumSubmissions IS NOT NULL AND MaximumSubmissions >= 1""".format(platform, package)), columns=['ProblemID'])['ProblemID'].tolist()
        problemsEvents = pd.DataFrame.from_records(db.execute("""SELECT AccountUserID, ProblemID, ProblemType, Title, Grade, SubmissionNumber, TimeStamp FROM {}.Problem_Events_with_Info WHERE DataPackageID = \'{}\' AND Grade IS NOT NULL""".format(platform, package)), columns=['AccountUserID', 'ProblemID', 'ProblemType', 'ProblemTitle', 'Grade', 'SubmissionNumber', 'GradeDate']).drop_duplicates()

        if len(problems) > 2 and None not in problems:
            ngrades = problemsEvents.groupby(by='ProblemID')['Grade'].max()

            problems = list(set(problems) & set(problemsEvents['ProblemID'].unique()))
            problemsEvents = problemsEvents[problemsEvents['ProblemID'].isin(problems)].sort_values(by=['AccountUserID', 'ProblemID', 'SubmissionNumber']).drop_duplicates(subset=['AccountUserID', 'ProblemID'], keep=config['keep'])

            if len(problemsEvents) > 0:
                nproblems = problemsEvents.groupby(by='AccountUserID').size()
                maxproblems = max(nproblems.values)
                filtered_user_ids = nproblems[nproblems == maxproblems].index.to_list()
                problemsEvents = problemsEvents[problemsEvents['AccountUserID'].isin(filtered_user_ids)].sort_values(by='GradeDate').reset_index(drop=True)
                problemsEvents['ProblemMax'] = maxproblems

                maxngrades = {k:v for k,v in zip(ngrades.index, ngrades.values)}
                problemsEvents['GradeMax'] = problemsEvents['ProblemID'].apply(lambda x: maxngrades[x])
                problemsEvents['GradeDate'] = problemsEvents['GradeDate'].apply(tmp2dt)

                y = problemsEvents[['AccountUserID', 'ProblemID', 'ProblemType', 'ProblemTitle', 'ProblemMax', 'Grade', 'GradeMax', 'GradeDate']]

                if 'aires-protegees' in course or 'EPFL-application-loi' in course or 'EPFL-conservation-especes' in course or 'suivi-eco' in course:
                    y.to_csv(os.path.join('labels', 'labels_' + course), index=False)
                print(course, len(problems), maxproblems, len(y['AccountUserID'].unique()))
            else:
                print('> skipped1', course)
        else:
            print('> skipped2', course)

    db.close()

def main():

    metadata = pd.read_csv('metadata.csv', encoding='iso-8859-1')

    config = {
        'courses': ['ca_courseware_EPFL-AlgebreLineaire-2018',
                    'ca_courseware_EPFL-AlgebreLineaire-2019',
                    'ca_courseware_EPFL-CS-210-2018_t3',
                    'ca_courseware_EPFL-CS-206-2019_T1']
    }

    downloadLabelsFromISA(config, metadata)

    config = {
        'keep': 'last'
    }

    #downloadQuizLabelsFromCoursera(config)

if __name__ == "__main__":
    main()

