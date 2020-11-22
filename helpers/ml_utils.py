#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
import numpy as np

def getTrainTestData(data, mode, task, ratio=0.9, thr=4.0):
    if task == 'binary':
        print('Binarizing student grades')
        y = [(1 if grade >= thr else 0) for grade in data['Grade']]
    elif task == 'multi-class':
        print('Rounding student grades')
        y = [round(grade) for grade in data['Grade']]
    else:
        raise NotImplementedError('The train-test split task ' + task + ' is not implemented.')

    print('>', [(c, y.count(c)) for c in np.unique(y)])

    if mode == 'random':
        print('Spitting the whole student population randomly')
        x_train, x_test, y_train, y_test = train_test_split(data['AccountUserID'], y, stratify=y, test_size=1 - ratio)
    elif mode == 'per-year':
        print('Spitting the whole student population per year')
        x_train = data[data['Round'] != sorted(data['Round'])[-1]]['AccountUserID']
        x_test = data[data['Round'] == sorted(data['Round'])[-1]]['AccountUserID']
        y_train = list(np.array(y)[np.array(data[data['Round'] != sorted(data['Round'])[-1]].index)])
        y_test = list(np.array(y)[np.array(data[data['Round'] == sorted(data['Round'])[-1]].index)])
    else:
        raise NotImplementedError('The train-test split mode ' + mode + ' is not implemented.')

    print('>', 'Train', len(y_train), [(c, y_train.count(c)) for c in np.unique(y_train)])
    print('>', 'Test', len(y_test), [(c, y_test.count(c)) for c in np.unique(y_test)])

    return x_train, x_test, y_train, y_test