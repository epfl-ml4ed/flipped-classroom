import json
import requests
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

#Script argument
parser = argparse.ArgumentParser(description='Srape youtube durations')
parser.add_argument('--video_csv', required=True) #path to the folder contaniing the 2 csv and xlsx
args = parser.parse_args()


def scrape_duration(url):
    video_id=url.split('watch?v=')[1]
    api_key="AIzaSyD0aLaWZoTMEGV3pcVvVrJR5oWVqw56vsk"
    url= "https://www.googleapis.com/youtube/v3/videos?id="+video_id+"&key="+api_key+"&part=contentDetails"
    source = requests.get(url).text
    data = json.loads(source)
    if len(data['items']) > 0:
        duration_str = data['items'][0]['contentDetails']['duration'] # Format PT{minutes}M{seconds}S
    else:
        return np.nan
    if "S" not in duration_str:
        duration_str += '0S'
    time = datetime.strptime(duration_str, 'PT%MM%SS').time()
    return time.minute * 60 + time.second 

def main():
    videos = pd.read_csv(args.video_csv, index_col=0)
    tqdm.pandas(desc="duration scraping")
    videos['Duration'] = videos.Source.progress_apply(lambda url: scrape_duration(url))
    videos.to_csv('video_with_durations.csv')

if __name__ == "__main__":
    main()
