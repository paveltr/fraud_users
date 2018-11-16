import numpy as np
import pandas as pd
import warnings
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import LSHForest
from sklearn.ensemble import IsolationForest
import gzip
import json
import logging
import pickle
import datetime
import os
import argparse

def clean_text(x, mode):
    """
    leaves only either numbers or letters
    """
    if mode == 'numbers':
        return ''.join(r for r in re.findall(r'[0-9]+', str(x)))
    elif mode == 'text':
        return ''.join(r for r in re.findall(r'[a-z]+', str(x).lower()))
    else:
        raise ValueError('choose correct mode value [numbers, text]')
        
        
def get_id(x):
    """
    clean up id field
    """
    return ''.join(r for r in re.findall(r'[0-9|a-zA-Z]+', str(x).split(':')[1])).strip()
    


def save_pickle(obj, name, path):
    """
    save objects in pickle format
    """
    with open(r'%s/%s.pkl' % (path, name), 'wb') as fid:
         pickle.dump(obj, fid, protocol=pickle.HIGHEST_PROTOCOL)
            
def load_pickle(name, path):
    """
    load objects in pickle format
    """
    with open(r'%s/%s.pkl' % (path, name), 'rb') as f:
        obj = pickle.load(f)
    return obj

    
def find_neighbors(distances, index_match):
    """
    checks if some user has neighbors in the specified radius
    """
    neigbors = []
    for d in distances[1]:
        if len(d) > 1:
            n = []
            for ii in d[1:]:
                n.append(index_match[ii])
            neigbors.append(n) 
        else:
            neigbors.append([])
    return neigbors