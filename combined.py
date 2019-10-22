#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 18:03:36 2019

@author: sonia
"""

import pandas as pd
import glob
import os

path = os.getcwd() + "/FrameObj/"

files = glob.gob(path + "*.txt")
Names = [f.split("_output")[0] for f in files]

for n in Names:
    listFiles = [f for f in os.listdir(path) if f.startswith(n)]
    combined = pd.concat([pd.read_csv(f) for f in listFiles])
    combined.to_csv()