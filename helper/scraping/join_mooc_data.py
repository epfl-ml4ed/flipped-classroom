#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import datetime
import argparse


### Concatenante the videos/problems csv (with the subchapters) with their due dates based on Subchapters
### given an excel spreadsheet containing the due dates and the videos and problems csv


#Script argument
parser = argparse.ArgumentParser(description='Joins video/problems with due dates based on SubCcapters.')
parser.add_argument('--year', required=True)
parser.add_argument('--input_folder', required=True) #path to the folder contaniing the 2 csv and xlsx
args = parser.parse_args()

PATH = args.input_folder
dates_url = PATH + 'due_dates_'+args.year+".xlsx"
problems_url = PATH + 'problems_'+args.year+".csv"
videos_url = PATH +'videos_'+args.year+".csv"


def merge_and_write(df, dates_df, suffix):
    dated_df = df.merge(dates_df, left_on="Subchapter", right_on="Subchapter")\
                # .sort_values(by="Subchapter")
    dated_df.to_csv(PATH+suffix)

def main():
    dates_df = pd.read_excel(dates_url,dtype={'Subchapter': str, 'Due_date': "datetime64"})
    problems_df = pd.read_csv(problems_url)
    videos_df = pd.read_csv(videos_url)
    merge_and_write(videos_df, dates_df, "dated_videos_"+args.year+".csv")
    merge_and_write(problems_df, dates_df, "dated_problems_"+args.year+".csv")

if __name__ == "__main__":
    main()

