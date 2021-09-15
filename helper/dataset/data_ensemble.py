"""
TODO: Check paths
"""

import re
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import json

def clean_text(x,etal):
    clean = {' ': '',
         '.': '_',
         '-': '_',
         ',': '',
         '__': '_',
         '(': '',
         ')': '',
        'number':'num',
        'videos':'vid',
        'duration':'dur',
        'distinct':'dist',
        'length':'len',
        'problem':'prob',
        'submission':'sub'
        }
    x = x.lower()
    for key in clean.keys():
        x = x.replace(key,clean[key])
    x = re.sub(r'<.+?>', 'fun', x)
    x = '{}_{}'.format(etal[:2],x)
    return x


def create_ensemble():
    courses = ['algebrelineaire_2018', 'algebrelineaire_2019', 'cs_206_2019_t1', 'cs_210_2018_t3']
    feature_authors = ['akpinar_et_al', 'lemay_doleck', 'mubarak_et_al',
    'boroujeni_et_al', 'he_et_al', 'marras_et_al',
    'chen_cui', 'lalle_conati', 'mbouzao_et_al', 'wan_et_al']
    weeks = ['eq', 'lq']


    for dataset in courses:
        for week in weeks:

            first = True
            all_feature_names = []

            for etal in feature_authors:
                dir_file = './../data/feature/{}_week-{}-epfl_{}'.format(week, etal, dataset)
                feat_values_dir = '{}/{}'.format(dir_file, 'feature_values.npz')
                data = np.load(feat_values_dir)
                lst = data.files

                sett_dir = '{}/{}'.format(dir_file, 'settings.txt')
                feat_settings = json.load(open(sett_dir, 'rb'))
                feature_names = feat_settings['feature_names']
                all_feature_names = all_feature_names + [clean_text(x,etal) for x in feature_names]

                labels_dir = '{}/{}'.format(dir_file, 'feature_labels.csv')
                sanity_check =  pd.read_csv(labels_dir)

                #Format data
                feat = data['feature_values']

                if first:
                    all_feat = feat
                    all_sanity_check = sanity_check
                    first = False
                else:
                    all_feat = np.concatenate((all_feat, feat), axis = 2)
                    if (np.sum(sanity_check == sanity_check).sum() == 5*len(sanity_check)) == False:
                        print('Error')


            etal = 'ensemble'
            dir_file = './../data/feature/{}_week-{}-epfl_{}'.format(week, etal, dataset)
            Path(dir_file).mkdir(parents=True, exist_ok=True)

            feat_values_dir = '{}/{}'.format(dir_file, 'feature_values.npz')
            np.savez(feat_values_dir, feature_values=all_feat)


            feat_settings['model'] = 'extractor.set.ensemble',
            feat_settings['feature_names'] = all_feature_names

            sett_dir = '{}/{}'.format(dir_file, 'settings.txt')
            with open(sett_dir, 'w') as file:
                file.write(json.dumps({**feat_settings}))


def create_ensamble_la():
    courses = ['algebrelineaire_2018', 'algebrelineaire_2019']
    weeks = ['eq', 'lq']


    for week in weeks:
        first = True
        for dataset in courses:
            print(dataset)
            etal = 'ensemble'

            dir_file = './../data/feature/{}_week-{}-epfl_{}'.format(week, etal, dataset)
            feat_values_dir = '{}/{}'.format(dir_file, 'feature_values.npz')
            data = np.load(feat_values_dir)
            lst = data.files

            sett_dir = '{}/{}'.format(dir_file, 'settings.txt')
            feat_settings = json.load(open(sett_dir, 'rb'))

            labels_dir =  './../data/labels/epfl_{}.csv'.format(dataset)
            feat_labels =  pd.read_csv(labels_dir)

            #Format data
            feat = data['feature_values']
            print(feat.shape)
            if first:
                all_feat = feat
                all_feat_labels = feat_labels
                first = False
            else:
                all_feat = np.concatenate((all_feat[:,:-1,:], feat), axis = 0)
                feat_labels['user_index'] = feat_labels['user_index'] + np.max(all_feat_labels['user_index']) + 1
                all_feat_labels = pd.concat([all_feat_labels, feat_labels])


        dataset = 'algebrelineaire'
        dir_file = './../data/feature/{}_week-{}-epfl_{}'.format(week, etal, dataset)
        Path(dir_file).mkdir(parents=True, exist_ok=True)

        feat_values_dir = '{}/{}'.format(dir_file, 'feature_values.npz')
        np.savez(feat_values_dir, feature_values=all_feat)

        feat_settings['course_id'] = dataset
        sett_dir = '{}/{}'.format(dir_file, 'settings.txt')
        with open(sett_dir, 'w') as file:
            file.write(json.dumps({**feat_settings}))

        labels_dir =  './../data/labels/epfl_{}.csv'.format(dataset)
        all_feat_labels.to_csv(labels_dir, index = False)
