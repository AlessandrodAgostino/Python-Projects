#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:36:33 2019

@author: alessandro
"""


import plumbum.cli.terminal as terminal
# see also ask and prompt
terminal.choose("favorite color?", ['red', 'green', 'blue'], default='blue')
#%%