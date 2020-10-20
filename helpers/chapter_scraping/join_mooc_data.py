#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import datetime
import argparse

#Script argument
parser = argparse.ArgumentParser(description='Parse a courseware xml folder.')
parser.add_argument('--year', required=True)
args = parser.parse_args()

PATH ='../MOOC/'+args.year+"/Data/"
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

