#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 17:07:13 2018

@author: sonia
"""

import pandas as pd
import glob


directoryPath = "/Users/sonia/Desktop/Projects/"
filenames = glob.glob(directoryPath+'*.csv')
combined_csv = pd.concat( [ pd.read_csv(f) for f in filenames ] )

combined_csv.to_csv( "combined_csv.csv", index=False )