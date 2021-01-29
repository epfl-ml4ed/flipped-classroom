#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from extractor.extractor import Extractor

class Ensemble(Extractor):

    def __init__(self, data, sets=None):
        assert sets is not None
        super().__init__('ensemble')
        self.data = data
        self.name = '-'.join([set.get_name() for set in sets])
        self.features = []
        for set in sets:
            self.features += set.get_features_objects()

    def get_features(self, data, settings):
        for f in self.features:
            f.set_data(data)
        features = [f.compute(settings) for f in self.features]
        assert len(features) == self.__len__()
        return np.nan_to_num(features)
