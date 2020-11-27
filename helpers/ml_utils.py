#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import os

def getTrainTestData(data, mode, task, ratio=0.9, thr=4.0):
    if task == 'binary':
        print('> Binarizing student grades:', end=' ')
        y = [(1 if grade >= thr else 0) for grade in data['Grade']]
    elif task == 'multi-class':
        print('> Rounding student grades:', end=' ')
        y = [round(grade) for grade in data['Grade']]
    else:
        raise NotImplementedError('The train-test split task ' + task + ' is not implemented.')

    print([(c, y.count(c)) for c in np.unique(y)])

    if mode == 'random':
        print('> Spitting the whole student population randomly:', end=' ')
        x_train, x_test, y_train, y_test = train_test_split(np.arange(len(data)), y, stratify=y, test_size=1 - ratio)
    elif mode == 'per-year':
        print('> Spitting the whole student population per year:', end=' ')
        x_train = data[data['Round'] != sorted(data['Round'])[-1]].index
        x_test = data[data['Round'] == sorted(data['Round'])[-1]].index
        y_train = list(np.array(y)[np.array(data[data['Round'] != sorted(data['Round'])[-1]].index)])
        y_test = list(np.array(y)[np.array(data[data['Round'] == sorted(data['Round'])[-1]].index)])
    else:
        raise NotImplementedError('The train-test split mode ' + mode + ' is not implemented.')

    print('Train', len(y_train), [(c, y_train.count(c)) for c in np.unique(y_train)], end = ' - ')
    print('Test', len(y_test), [(c, y_test.count(c)) for c in np.unique(y_test)])

    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

def loadFeatureSets(mode, task, ratio, start, end, step):
    filename = '../data/feature_sets/feature_sets_' + mode + '_' + task + '_' + str(ratio) + '_' + str(start) + '-' + str(end) + '-' + str(step) + '.pkl'

    if os.path.exists(filename):
        print('> Found features for this experimental setting in', filename)
        with open(filename, 'rb') as f:
            feature_sets = pickle.load(f)
    else:
        print('> Initialized features for this experimental setting in', filename)
        feature_sets = {}

    return feature_sets

def saveFeatureSets(feature_sets, mode, task, ratio, start, end, step):
    filename = '../data/feature_sets/feature_sets_' + mode + '_' + task + '_' + str(ratio) + '_' + str(start) + '-' + str(end) + '-' + str(step) + '.pkl'

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    with open(filename, 'wb') as f:
        print('> Saved features for this experimental setting in', filename)
        pickle.dump(feature_sets, f, protocol=pickle.HIGHEST_PROTOCOL)

def computeFeatures(event_data, exam_data, feature_labels, feature_sets, start, weeks):

    for findex, ffunc in enumerate(feature_labels):
        flabel = ffunc.getName()

        if not flabel in feature_sets:
            feature_sets[flabel] = {}

            for windex, wid in enumerate(weeks):
                feature_sets[flabel][wid] = []

                unactiveTrain = 0
                for uindex, uid in enumerate(exam_data['AccountUserID']):

                    udata = event_data[(event_data['AccountUserID'] == uid) & (event_data['Week'] >= start) & (event_data['Week'] < wid)]
                    year = int(exam_data[exam_data['AccountUserID'] == uid]['Round'].values[0].split('-')[-2])

                    if len(udata) > 0:
                        feature_sets[flabel][wid].append(ffunc.getUserFeatures(udata, wid, year))
                    else:
                        unactiveTrain += 1
                        feature_sets[flabel][wid].append([0 for i in range(ffunc.getNbFeatures())])

                    if (uindex + 1) % 10 == 0 or (uindex + 1) == len(exam_data['AccountUserID']):
                        print('\r> Set from week', start, ':', flabel, '(', '{:03d}'.format(findex + 1), '{:03d}'.format(len(feature_sets)), ')', end=' - ')
                        print('Week:', '{:03d}'.format(wid), '(', '{:03d}'.format(windex + 1), '{:03d}'.format(len(weeks)), ')', end=' - ')
                        print('User:', '{:03d}'.format(uindex + 1), '{:03d}'.format(len(exam_data['AccountUserID'])), end='')

                feature_sets[flabel][wid] = np.array(feature_sets[flabel][wid])

    print()

    return feature_sets


def loadTrainedModels(mode, task, ratio, start, end, step):
    filename = '../data/trained_models/trained_models_' + mode + '_' + task + '_' + str(ratio) + '_' + str(start) + '-' + str(end) + '-' + str(step) + '.pkl'

    if os.path.exists(filename):
        print('> Found models for this experimental setting in', filename)
        with open(filename, 'rb') as f:
            trained_models = pickle.load(f)
    else:
        print('> Initialized models for this experimental setting in', filename)
        trained_models = {}

    return trained_models

def trainModels(feature_sets, x_train, y_train, weeks, classifiers_types, classifiers_params, trained_models, isScaled=True):
    scaler = [StandardScaler() if isScaled else None for _ in range(len(feature_sets))]

    for findex, flabel in enumerate(feature_sets.keys()):

        if not flabel in trained_models:
            trained_models[flabel] = {}

        for windex, wid in enumerate(weeks):

            if not wid in trained_models[flabel]:
                trained_models[flabel][wid] = {}

            for mindex, mid in enumerate(classifiers_types.keys()):

                if not mid in trained_models[flabel][wid]:
                    trained_models[flabel][wid][mid] = []

                print('\r> Training on Set:', flabel, '(', '{:03d}'.format(findex + 1), '{:03d}'.format(len(feature_sets)), ')', end=' - ')
                print('Week:', '{:03d}'.format(wid), '(', '{:03d}'.format(windex + 1), '{:03d}'.format(len(weeks)), ')', end=' - ')
                print('Algorithm:', mid.ljust(3), '(', '{:03d}'.format(mindex + 1), '{:03d}'.format(len(classifiers_types)), ')', end='')

                clf = GridSearchCV(classifiers_types[mid], classifiers_params[mid])

                if isScaled:
                    clf.fit(scaler[findex].fit_transform(feature_sets[flabel][wid][x_train]), y_train)
                else:
                    clf.fit(feature_sets[flabel][wid][x_train], y_train)

                trained_models[flabel][wid][mid].append(clf)

    print()

    return trained_models, scaler

def showBestParams(trained_models):
    for flabel, fvalue in trained_models.items():
        print('Feature set:', flabel)
        for wid, wvalue in fvalue.items():
            print('> Week:', wid)
            for mid, mvalue in wvalue.items():
                print('>> Model:', mid, '-', trained_models[flabel][wid][mid].best_params_)

def saveTrainedModels(trained_models, mode, task, ratio, start, end, step):
    filename = '../data/trained_models/trained_models_' + mode + '_' + task + '_' + str(ratio) + '_' + str(start) + '-' + str(end) + '-' + str(step) + '.pkl'

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    with open(filename, 'wb') as f:
        print('> Saved models for this experimental setting in', filename)
        pickle.dump(trained_models, f, protocol=pickle.HIGHEST_PROTOCOL)

def tn(y_true, y_pred):
    return confusion_matrix(y_true, y_pred).ravel()[0]

def fp(y_true, y_pred):
    return confusion_matrix(y_true, y_pred).ravel()[1]

def fn(y_true, y_pred):
    return confusion_matrix(y_true, y_pred).ravel()[2]

def tp(y_true, y_pred):
    return confusion_matrix(y_true, y_pred).ravel()[3]

def tpr(y_true, y_pred):
    return tp(y_true, y_pred) / (tp(y_true, y_pred) + fn(y_true, y_pred))

def tnr(y_true, y_pred):
    return tn(y_true, y_pred) / (tn(y_true, y_pred) + fp(y_true, y_pred))

def fpr(y_true, y_pred):
    return fp(y_true, y_pred) / (fp(y_true, y_pred) + tn(y_true, y_pred))

def fnr(y_true, y_pred):
    return fn(y_true, y_pred) / (tp(y_true, y_pred) + fn(y_true, y_pred))

def eer(y_true, y_pred):
    return np.mean([fnr(y_true, y_pred), fpr(y_true, y_pred)])

def computeMetrics(scaler, feature_sets, trained_models, x_test, y_test, evaluation_metrics, evaluation_scores):

    for findex, (flabel, fvalue) in enumerate(trained_models.items()):

        if not flabel in evaluation_scores:
            evaluation_scores[flabel] = {}

        for windex, (wid, wvalue) in enumerate(trained_models[flabel].items()):

            if not wid in evaluation_scores[flabel]:
                evaluation_scores[flabel][wid] = {}

            test_data = scaler[findex].transform(feature_sets[flabel][wid][x_test])

            for mindex, (mid, mvalue) in enumerate(trained_models[flabel][wid].items()):

                if not mid in evaluation_scores[flabel][wid]:
                    evaluation_scores[flabel][wid][mid] = {}

                for emid, mfunc in evaluation_metrics.items():

                    if not emid in evaluation_scores[flabel][wid][mid]:
                        evaluation_scores[flabel][wid][mid][emid] = []

                    clf = trained_models[flabel][wid][mid][-1]
                    evaluation_scores[flabel][wid][mid][emid].append(mfunc(y_test, np.around(clf.predict(test_data))))

    print('> Evaluated last iteration models')

    return evaluation_scores