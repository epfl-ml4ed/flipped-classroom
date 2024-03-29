#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_grade_distribution(course, mingrade=1, maxgrade=6, depgrade=.25, thrgrade=4.0):

    grades = np.arange(mingrade, maxgrade + 2 * depgrade, depgrade)
    x = course.get_clickstream_grade()['grade'].values

    # Histogram
    N, bins, patches = plt.hist(x, density=True, bins=grades, edgecolor='white', label='Data')
    bin_w = (max(grades) - min(grades)) / (len(grades) - 1)
    plt.xticks(np.arange(min(grades) + bin_w / 2, max(grades), bin_w), grades[:-1], rotation=45)

    for i in range(0, len(grades[grades < thrgrade])):
        patches[i].set_facecolor('#D16666')
    for i in range(len(grades[grades < thrgrade]), len(patches)):
        patches[i].set_facecolor('#50A2A7')

    # Line distribution
    mn, mx = plt.xlim()
    plt.xlim(mn, mx)
    kde_xs = np.linspace(mn, mx, 300)
    kde = gaussian_kde(x)
    plt.plot(kde_xs, kde.pdf(kde_xs), label='PDF', color='#000000', linestyle='--')

    # Line vertical
    plt.axvline(x=thrgrade, color='#818D92', linestyle='--')

    # Decoration
    plt.title(r'$\bf{Grade}$ $\bf{Distribution}$')
    plt.ylabel('Probability')
    plt.xlabel('Final Grade')
    plt.xlim([mingrade, maxgrade + depgrade])
    plt.ylim([0.0, 1.0])
    plt.grid(axis='y')

def plot_pass_fail_distribution(course):
    x = course.get_clickstream_grade()['label-pass-fail'].to_list()

    # Bar Plot
    plt.bar(['Pass', 'Fail'], [x.count(0) / len(x), x.count(1) / len(x)], color=['#50A2A7', '#D16666'], width=0.4, edgecolor='white')

    # Decoration
    plt.title(r'$\bf{Pass-Fail}$ $\bf{Distribution}$')
    plt.ylabel('Percentage')
    plt.xlabel('Student Category')
    plt.ylim([0, 1])
    plt.grid(axis='y')

def plot_dropout_distribution(course):
    x = course.get_clickstream_grade()['label-dropout'].to_list()

    # Bar Plot
    plt.bar(['Stay', 'Drop'], [x.count(0) / len(x), x.count(1) / len(x)], color=['#50A2A7', '#D16666'], width=0.4, edgecolor='white')

    # Decoration
    plt.title(r'$\bf{Dropout}$ $\bf{Distribution}$')
    plt.ylabel('Percentage')
    plt.xlabel('Student Category')
    plt.ylim([0, 1])
    plt.grid(axis='y')

def plot_stopout_distribution(course, extra_weeks=8):
    x = course.get_clickstream_grade()['label-stopout'].to_list()

    w = np.arange(course.get_weeks() + extra_weeks + 1)
    d = np.array([x.count(el) for el in w])

    # Histogram
    plt.bar(w, d / np.sum(d), width=0.4, edgecolor='white', color='#D16666')
    plt.xticks(w, w, rotation=45)

    # Line vertical
    plt.axvline(x=course.get_weeks() - 1, color='#818D92', linestyle='--')

    # Decoration
    plt.title(r'$\bf{Stopout}$ $\bf{Distribution}$')
    plt.ylabel('Percentage')
    plt.xlabel('Stopout Week')
    plt.xlim([0, course.get_weeks() + extra_weeks + .5])
    plt.ylim([0.0, 1.0])
    plt.grid(axis='y')

def plot_feature(feature, feature_values, groups):
    plt.figure(figsize=(30, 8), dpi=300)
    for label, idxs, color in groups:
        plt.plot(np.arange(feature_values.shape[1]), np.nanmean(feature_values[idxs, :], axis=0), label=label, color=color)
    plt.title(feature)
    plt.ylabel(feature)
    plt.xlabel('week')
    plt.xlim(0, len(np.arange(feature_values.shape[1])[:-2]))
    plt.ylim([0, 1])
    plt.grid()
    plt.legend()

def plot_feature_per_model(timeframe, target, course_id, metric='f1', ylim=np.array([0.0, 1.0]), filepath='../data/result/edm21/predictor'):
    predictors = set([p.split('-')[3] for p in os.listdir(filepath) if target in p and course_id in p and not 'ensemble' in p])
    results = [p for p in os.listdir(filepath) if course_id in p and target in p and not 'ensemble' in p]

    plt.figure(figsize=(30, 24), dpi=300)
    plt.rcParams.update({'font.size': 16})
    for p_idx, predictor in enumerate(predictors):
        plt.subplot(3, 2, p_idx + 1)
        plt.title(predictor)
        for predictor_result in [predictor_result for predictor_result in results if '-' + predictor + '-' in predictor_result]:
            data_with_folds = pd.read_csv(os.path.join(filepath, predictor_result, 'stats.csv'))[['week', 'fold', metric]]
            data_per_fold = data_with_folds.groupby(by='week').mean(metric)
            plt.plot(data_per_fold.index, data_per_fold[metric], label=predictor_result.split('-')[4])
            plt.xlim([data_per_fold.index.min(), data_per_fold.index.max()])
            plt.ylabel(metric)
            plt.xlabel('week')
            plt.ylim(ylim)
            plt.grid(linewidth=1)
        plt.legend(loc='upper left', ncol=4)
    plt.tight_layout()
    plt.show()

def plot_feature_per_model_at_timeframe(predictor, target, course_id, metric='f1', ylim=np.array([0.0, 1.0]), filepath='../data/result/edm21/predictor'):
    features = set([p.split('-')[4] for p in os.listdir(filepath) if target in p and predictor in p and course_id in p and not 'ensemble' in p])
    results = [p for p in os.listdir(filepath) if target in p and predictor in p and course_id in p and 'eq_week' in p and not 'ensemble' in p]

    plt.figure(figsize=(30, int(np.round(len(features) // 5, 0)) * 8), dpi=300)
    plt.rcParams.update({'font.size': 16})
    for f_idx, result in enumerate(results):
        lq_week = [p for p in os.listdir(filepath) if result.split('-')[4] in p and target in p and predictor in p and course_id in p and 'lq_week' in p]

        if not len(lq_week) == 1:
            continue

        plt.subplot(int(np.round(len(features) // 5, 0)), 5, f_idx + 1)
        plt.title(result.split('-')[4])

        data_with_folds = pd.read_csv(os.path.join(filepath, result, 'stats.csv'))[['week', 'fold', metric]]
        data_per_fold = data_with_folds.groupby(by='week').mean(metric)
        plt.plot(data_per_fold.index, data_per_fold[metric], label='eq_week')

        data_with_folds = pd.read_csv(os.path.join(filepath, lq_week[0], 'stats.csv'))[['week', 'fold', metric]]
        data_per_fold = data_with_folds.groupby(by='week').mean(metric)
        plt.plot(data_per_fold.index, data_per_fold[metric], label='lq_week')

        plt.xlim([data_per_fold.index.min(), data_per_fold.index.max()])
        plt.ylabel(metric)
        plt.xlabel('week')
        plt.ylim(ylim)
        plt.grid(linewidth=1)
        plt.legend(loc='upper left', ncol=2)
    plt.tight_layout()
    plt.show()

def plot_feature_vs_ensemble(timeframe, target, course_id, metric='f1', ylim=np.array([0.0, 1.0]), filepath='../data/result/edm21/predictor'):
    predictors = set([p.split('-')[3] for p in os.listdir(filepath) if target in p and course_id in p and 'ensemble' in p and ('vec' in p or 'none' in p)])
    results = [p for p in os.listdir(filepath) if target in p and course_id in p and 'ensemble' in p and ('vec' in p or 'none' in p)]

    plt.figure(figsize=(30, 10), dpi=300)

    plt.rcParams.update({'font.size': 16})
    for p_idx, predictor in enumerate(predictors):
        print(p_idx, predictor)
        predictor_result = [predictor_result for predictor_result in results if '-' + predictor + '-' in predictor_result][0]
        data_with_folds = pd.read_csv(os.path.join(filepath, predictor_result, 'stats.csv'))[['week', 'fold', metric]]
        data_per_fold = data_with_folds.groupby(by='week').mean(metric)
        plt.plot(data_per_fold.index, data_per_fold[metric], label=predictor)
        plt.xlim([data_per_fold.index.min(), data_per_fold.index.max()])
        plt.ylabel(metric)
        plt.xlabel('week')
        plt.ylim(ylim)
        plt.grid(linewidth=1)
    plt.legend(loc='upper left', ncol=4)
    plt.tight_layout()
    plt.show()
