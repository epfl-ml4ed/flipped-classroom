#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def import_class(name):
    mod = __import__('.'.join(name.split('.')[:-1]), fromlist=[name.split('.')[-1]])
    return getattr(mod, name.split('.')[-1])