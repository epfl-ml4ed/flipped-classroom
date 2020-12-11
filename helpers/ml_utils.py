#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import os

def getTrainTestData(data, mode, task, ratio=0.9, thr=4.0, random_state=None):
    if task == 'binary':
        print('> Binarizing student grades:', end=' ')
        y = [(0 if grade >= thr else 1) for grade in data['Grade']] # 1 for students that failed (grade < 4)
    elif task == 'multi-class':
        print('> Rounding student grades:', end=' ')
        y = [round(grade) for grade in data['Grade']]
    else:
        raise NotImplementedError('The train-test split task ' + task + ' is not implemented.')

    print([(c, y.count(c)) for c in np.unique(y)])

    if mode == 'random':
        print('> Splitting the whole student population randomly:', end=' ')
        x_train, x_test, y_train, y_test = train_test_split(np.arange(len(data)), y, stratify=y, test_size=1 - ratio, random_state=random_state)
    elif mode == 'per-year':
        print('> Splitting the whole student population per year:', end=' ')
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

def computeFeatures(event_data, exam_data, feature_labels, feature_sets, start, weeks, weekly=False):
    """
    :param weekly: if set to True, compute the features for each week independantly, if False compute the
    features from the start until each week.
    """
    for findex, ffunc in enumerate(feature_labels):
        flabel = ffunc.getName()

        if not flabel in feature_sets:
            feature_sets[flabel] = {}

            for windex, wid in enumerate(weeks):
                feature_sets[flabel][wid] = []

                unactiveTrain = 0
                for uindex, uid in enumerate(exam_data['AccountUserID']):
                    if not weekly:
                        udata = event_data[(event_data['AccountUserID'] == uid) & (event_data['Week'] >= start) & (event_data['Week'] < wid)]
                    else:
                        udata = event_data[(event_data['AccountUserID'] == uid) & (event_data['Week'] == wid - 1)]
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


def loadTrainedModels(name, mode, task, ratio, start, end, step):
    filename = '../data/trained_models/trained_models_'+ name + '_' + mode + '_' + task + '_' + str(ratio) + '_' + str(start) + '-' + str(end) + '-' + str(step) + '.pkl'

    if os.path.exists(filename):
        print('> Found models for this experimental setting in', filename)
        with open(filename, 'rb') as f:
            trained_models, scaler, random_state = pickle.load(f)
    else:
        print('> Initialized models for this experimental setting in', filename)
        trained_models, scaler, random_state = {}, {}, np.random.randint(0,1e5)

    return trained_models, scaler, random_state


def saveTrainedModels(trained_models, name, mode, task, ratio, start, end, step, scaler, random_state):
    filename = '../data/trained_models/trained_models_'+ name + '_' + mode + '_' + task + '_' + str(ratio) + '_' + str(start) + '-' + str(end) + '-' + str(step) + '.pkl'

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    with open(filename, 'wb') as f:
        print('> Saved models for this experimental setting in', filename)
        pickle.dump((trained_models, scaler, random_state), f, protocol=pickle.HIGHEST_PROTOCOL)

    

def trainModels(feature_sets, x_train, y_train, weeks, classifiers_types, classifiers_params, trained_models, isScaled=True, x_test=None, y_test=None):
    scaler = {} #[[StandardScaler() for _ in weeks] if isScaled else None for _ in range(len(feature_sets))]

    for findex, flabel in enumerate(feature_sets.keys()):
        scaler[flabel] = {}
        if not flabel in trained_models:
            trained_models[flabel] = {}

        for windex, wid in enumerate(weeks):
            scaler[flabel][wid] = StandardScaler()
            if not wid in trained_models[flabel]:
                trained_models[flabel][wid] = {}

            if isScaled:
                data_train = scaler[flabel][wid].fit_transform(feature_sets[flabel][wid][x_train])
                if x_test!= None and y_test != None:
                    data_test = scaler[flabel][wid].transform(feature_sets[flabel][wid][x_test])
            else:
                data_train = feature_sets[flabel][wid][x_train]
                if x_test!= None and y_test != None:
                    data_test = feature_sets[flabel][wid][x_test]
                
            for mindex, mid in enumerate(classifiers_types.keys()):

                if not mid in trained_models[flabel][wid]:
                    trained_models[flabel][wid][mid] = []
                
                print('Week:', '{:03d}'.format(wid), end=' - ')
                print('Algorithm:', mid.ljust(3), end=' ')
                
                clf = GridSearchCV(classifiers_types[mid], classifiers_params[mid], scoring='f1')
                clf.fit(data_train, y_train)
                trained_models[flabel][wid][mid].append(clf)

                print('Training: {:.2f}'.format(accuracy_score(y_train, np.around(clf.predict(data_train)))), end=' ')
                if x_test!= None and y_test != None:
                    print('Test: {:.2f}'.format(accuracy_score(y_test, np.around(clf.predict(data_test)))), end=' ')
                print(f'CV best: {clf.best_score_:.2f}', end=' ')
                
                if mid == 'las':    #If the model is lasso then print the number of non zero coefficients
                    print((np.array(list(clf.best_estimator_.coef_[0])) != 0).sum())
                else:
                    print()
    print()
    return trained_models, scaler

def showBestParams(trained_models):
    for flabel, fvalue in trained_models.items():
        print('Feature set:', flabel)
        for wid, wvalue in fvalue.items():
            print('> Week:', wid)
            for mid, mvalue in wvalue.items():
                print('>> Model:', mid, '-', trained_models[flabel][wid][mid][0].best_params_)


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

            test_data = scaler[flabel][wid].transform(feature_sets[flabel][wid][x_test])

            for mindex, (mid, mvalue) in enumerate(trained_models[flabel][wid].items()):

                if not mid in evaluation_scores[flabel][wid]:
                    evaluation_scores[flabel][wid][mid] = {}

                for emid, mfunc in evaluation_metrics.items():

                    if not emid in evaluation_scores[flabel][wid][mid]:
                        evaluation_scores[flabel][wid][mid][emid] = []

                    clf = trained_models[flabel][wid][mid]
                    evaluation_scores[flabel][wid][mid][emid].append(mfunc(y_test, np.around(clf.predict(test_data))))

    print('> Evaluated last iteration models')

    return evaluation_scores


def fitWithBestHyperparamters(trainedModels, scaler, featureSets, x_train, y_train):
    """
    Compare scores over different training-test splits in order to 
    select the right hyperparamter combination for each week.
    Then train the models on the training set with the chosen hyperparameters.
    """
    bestModels = {}
    for flabel in trainedModels.keys():
        bestModels[flabel] = {}
        
        for windex, wid in enumerate(trainedModels[flabel].keys()):
            bestModels[flabel][wid] = {}
            
            for mindex, mid in enumerate(trainedModels[flabel][wid].keys()):
                #List of different models trained with different splits
                models = trainedModels[flabel][wid][mid] 

                #Sum the scores for each hyperparameter combinations (mean_test_score is the list 
                #of average split scores during the trainModel cross validation)
                #I.e., the CV performed in trainModel is not used to select the best hyperparamters 
                #but to compute scores for each hyperparamters combinations that are then averaged
                #over different train/test split

                scores = models[0].cv_results_['mean_test_score'] 
                for model in models[1:]: #Each model has been fitted on a different train/test split
                    scores += model.cv_results_['mean_test_score'] 
                
                #Select the best hyperparameter combination in average
                hyperparameters = model.cv_results_['params'][np.nanargmax(scores)]
                
                #Initialize and fit a new model with the selected hyperparameters
                clf = model.estimator.__class__().set_params(**hyperparameters)
                
                data_train = scaler[flabel][wid].transform(featureSets[flabel][wid][x_train])
                bestModels[flabel][wid][mid] = clf.fit(data_train, y_train)
    return bestModels