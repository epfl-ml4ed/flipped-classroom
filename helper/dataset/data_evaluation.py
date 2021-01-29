#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.metrics import confusion_matrix
import numpy as np

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