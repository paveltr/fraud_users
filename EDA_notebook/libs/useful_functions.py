import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
import re
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import LSHForest
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

import keras
from keras.models import Model

from itertools import combinations

import networkx as nx
import gzip
import json

sns.set(rc={'figure.figsize':(10,8)})
plt.style.use('seaborn-colorblind')
plt.rcParams.update({'grid.color' : '#53868B'})
plt.rcParams.update({'grid.linewidth' : 0})
plt.rcParams.update({'axes.facecolor' : '#53868B'})
plt.rcParams.update({'lines.linewidth' : 3})
plt.rcParams.update({'axes.prop_cycle' : plt.cycler(color=plt.cm.Set3.colors)})

plt.rcParams.update({'font.size' : 16})
plt.rcParams.update({'axes.titlesize': 16})
plt.rcParams.update({'legend.fontsize': 16})
plt.rcParams.update({'xtick.labelsize': 14})
plt.rcParams.update({'ytick.labelsize': 14})
warnings.filterwarnings('ignore')


def clean_text(x, mode):
    if mode == 'numbers':
        return ''.join(r for r in re.findall(r'[0-9]+', str(x)))
    elif mode == 'text':
        return ''.join(r for r in re.findall(r'[a-z]+', str(x).lower()))
    else:
        raise ValueError('choose correct mode value [numbers, text]')