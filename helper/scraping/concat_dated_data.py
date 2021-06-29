#!/usr/bin/env python
# coding: utf-8

import os.path
import pandas as pd

years = ['2017', '2018', '2019']

def concat_files(video=True, output_suffix="concat_"):
    """
    :param video: True to concatenate videos csv, False to concat problems csv 
    """
    
    concatenated_files =[]
    file_sizes = []
    if video:
        cols =['Chapter', 'Subchapter', 'VideoID','Source','Due_date']
    else:
        cols =['Chapter', 'Subchapter', 'ProblemID','Due_date']
    concat_df = pd.DataFrame(columns=cols)
    
    for year in years:
        filename = "dated_" + ("videos" if(video) else "problems") + "_" + year +".csv"
        path = './' + year + '/Data/' + filename
        if os.path.isfile(path):
            concatenated_files.append(filename)
            curr_df = pd.read_csv(path, index_col=0)
            file_sizes.append(str(len(curr_df)))
            concat_df = pd.concat([concat_df, curr_df])
        
    #reset the indices since they are not unique anymore
    concat_df.reset_index(drop=True, inplace=True)
        
    print("Concatenated files: {}".format(", ".join(concatenated_files)))
    print("Total length: {} rows, from file sizes: {}".format(len(concat_df), ", ".join(file_sizes)))
    


    output = output_suffix + ("videos" if(video) else "problems") + ".csv"
    concat_df.to_csv("./" + output)

def main():
    concat_files(video=True)
    concat_files(video=False)

if __name__ == "__main__":
    main()



