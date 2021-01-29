#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Feature():

    def __init__(self, name):
        self.name = name

    def compute(self):
        return None

    def get_name(self):
        return self.name

    def set_data(self, data):
        self.data = data