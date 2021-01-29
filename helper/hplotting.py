#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np

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
    x = course.get_clickstream_grade()['pass-fail'].to_list()

    # Bar Plot
    plt.bar(['Pass', 'Fail'], [x.count(0) / len(x), x.count(1) / len(x)], color=['#50A2A7', '#D16666'], width=0.4, edgecolor='white')

    # Decoration
    plt.title(r'$\bf{Pass-Fail}$ $\bf{Distribution}$')
    plt.ylabel('Percentage')
    plt.xlabel('Student Category')
    plt.ylim([0, 1])
    plt.grid(axis='y')

def plot_dropout_distribution(course):
    x = course.get_clickstream_grade()['dropout'].to_list()

    # Bar Plot
    plt.bar(['Stay', 'Drop'], [x.count(0) / len(x), x.count(1) / len(x)], color=['#50A2A7', '#D16666'], width=0.4, edgecolor='white')

    # Decoration
    plt.title(r'$\bf{Dropout}$ $\bf{Distribution}$')
    plt.ylabel('Percentage')
    plt.xlabel('Student Category')
    plt.ylim([0, 1])
    plt.grid(axis='y')

def plot_stopout_distribution(course, extra_weeks=8):
    x = course.get_clickstream_grade()['stopout'].to_list()

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
    for label, idxs, color in groups:
        plt.plot(np.arange(feature_values.shape[1]), np.mean(feature_values[idxs, :], axis=0), label=label, color=color)
    plt.title(feature.get_name())
    plt.ylabel(feature.get_name())
    plt.xlabel('week')
    plt.xlim(0, len(np.arange(feature_values.shape[1])) - 1)
    plt.grid()
    plt.legend()