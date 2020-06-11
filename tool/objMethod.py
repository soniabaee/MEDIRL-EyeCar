# -*- coding: utf-8 -*-
"""
	saving/loading an object as a pickle
"""

import pickle

class ObjMethod:
    
    def __init__(self,dirPath):
        self.dir = dirPath
        
    def save_obj(self, obj, name):
        with open(self.dir + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
            
    def load_obj(self, name):
        with open(self.dir + name + '.pkl', 'rb') as f:
            return pickle.load(f)