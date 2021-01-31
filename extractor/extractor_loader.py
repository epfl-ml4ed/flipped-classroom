#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import json
import os

from extractor.extractor import Extractor

class ExtractorLoader(Extractor):

    def __init__(self):
        super().__init__('extractor_loader')

    def save(self, course, settings):
        raise NotImplementedError()

    def extract_features(self, data, settings):
        raise NotImplementedError()

    def extract_features_bunch(self, course, settings):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()