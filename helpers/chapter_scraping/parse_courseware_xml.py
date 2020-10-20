#!/usr/bin/env python
# coding: utf-8

import xml.etree.ElementTree as ET
from xml.dom import minidom
import re
import csv
import argparse

#Script argument
parser = argparse.ArgumentParser(description='Parse a courseware xml folder.')
parser.add_argument('--year', required=True)
args = parser.parse_args()

#Constants
PATH = "../MOOC/"+ args.year +"/courseware_xml/"
ORIGIN = 'course.xml'

#XML commom attributes
URL = 'url_name'
NAME = 'display_name'
YT_URL = 'youtube_id_1_0'

#XML elements
ELEM_TO_PARSE = ['course','chapter','sequential','vertical','problem','video'] #ignore other elements
END_RECUR = ['problem','video'] #stop the recursion on these elements

YT_PREFIX = "https://www.youtube.com/watch?v="

videos = [] # videos = [[chapter, subchapter, url_name, youtube_url], ...]
problems = [] # problems = [[chapter, subchapter, url_name], ...]


def extract_subchapter(seq):
    title =  seq.attrib[NAME]
    first_word = title.split(' ')[0]
    if re.search("^([0-9]*[.]?)+$",first_word):
        return first_word 
    else:
        return str(0) #return subchapter as a string 


def read_node(tag, url, subchapter="0"):
    curr_tree = ET.parse("{path}/{tag}/{url}.xml".format(path=PATH, tag=tag, url=url))
    curr_root = curr_tree.getroot()

    if curr_root.tag == "sequential":
        subchapter = extract_subchapter(curr_root)
        
    if curr_root.tag == 'video':
        videos.append([subchapter.split('.')[0], subchapter, url,  YT_PREFIX + curr_root.attrib[YT_URL]])
        
    if curr_root.tag == 'problem':
        problems.append([subchapter.split('.')[0], subchapter, url])
        
    if curr_root.tag not in END_RECUR: #Stop recursion on Problems and Videos
        for child in curr_root:
            if child.tag in ELEM_TO_PARSE: #Parse only tags of interest
                read_node(child.tag, child.attrib[URL], subchapter)


def write_csv(videos, problems, year):
    write_path = "../MOOC/" + args.year +"/Data/" 
    with open(write_path + "videos_"+year+".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([["Chapter", "Subchapter", "VideoID", "Source"]])
        writer.writerows(videos)
    with open(write_path + "problems_"+year+".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([["Chapter","Subchapter", "ProblemID"]])
        writer.writerows(problems)


def main():
    tree = ET.parse(PATH + ORIGIN)
    root = tree.getroot()
    read_node(root.tag, root.attrib[URL])
    write_csv(videos,problems, args.year)


if __name__ == "__main__":
    main()

