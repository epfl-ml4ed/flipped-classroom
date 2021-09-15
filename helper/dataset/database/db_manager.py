#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from .hsql import queryDB

# Check this
SQL_DIR = Path().resolve().parent / 'helper/dataset/database/sql'
CSV_DIR = Path().resolve().parent / 'data/course'

class DBManager():

    def __init__(self, id, type, platform):
        self.course_id = id
        self.type = type
        self.platform = platform

    def prepare_data(self):
        # Forum
        forum_name = 'forum_event'
        forum_cols =  ['user_id', 'forum_id', 'event_type', 'timestamp',
                        'post_type', 'post_title', 'post_text']
        self.create_csv(forum_name, forum_cols)


    def exists(self, path):
        print("Already exists")
        return os.path.exists(path)

    def create_csv(self, event_name, col_names):
        output_dir = CSV_DIR / self.type / self.platform / event_name
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_file = output_dir / '{}.csv'.format(self.course_id)
        print(output_file)
        if self.exists(output_file):
            return

        if self.platform in ('courseware'):
            event_name =  event_name  + '_hash'

            input_file = SQL_DIR / '{}.sql'.format(event_name)
            print(input_file)
            if os.path.exists(input_file):
                with open(input_file, 'r',encoding="utf-8") as file:
                    sql_query = file.read()
                formatted_query = sql_query.format(course = self.course_id,
                                                   platform = self.platform)
                df = queryDB(formatted_query, col_names)
                df.to_csv(output_file, index = False)

        elif self.platform in ('coursera'):
            df = self.create_csv_large()
            df.columns = col_names
            df.to_csv(output_file, index = False)


    def create_csv_large(self):
        query = """
            SELECT AccountUserID, CONVERT(SessionUserID USING utf8) AS user_id
            FROM ca_{platform}.User_Hash_Mapping
            WHERE DataPackageID IN ('{course}');
        """.format(course = self.course_id, platform = self.platform)
        df_codes = queryDB(query)

        query = """
            SELECT * FROM ca_{platform}.Forum_Events
            WHERE DataPackageID IN ('{course}');
        """.format(course = self.course_id, platform = self.platform)
        df_events = queryDB(query)

        query = """
            SELECT PostText,DataPackageID,PostType,PostID, PostTitle
            FROM ca_{platform}.Forum_Info
            WHERE DataPackageID IN ('{course}');
        """.format(course = self.course_id, platform = self.platform)
        df_info = queryDB(query)

        df_events_info = pd.merge(df_events, df_info,  how='left')
        df_merged = pd.merge(df_events_info, df_codes, how = 'left')

        selected_cols= ['user_id', 'PostID', 'EventType', 'TimeStamp','PostType',
                        'PostTitle','PostText']
        df = df_merged[selected_cols]
        return df
