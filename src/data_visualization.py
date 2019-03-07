import os
import re
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

path = '../data/'
fig_path = '../plots/data/'
max_buckets = 20

files = [f for f in os.listdir(path) if f.find('_') < 0]

for filename in files:
    p = os.path.join(path, filename)
    df = pd.read_csv(p)
    cols = df.columns
    classes = df['class'].unique()

    directory = os.path.join(fig_path, filename.split('.')[0])
    if not os.path.exists(directory):
        os.makedirs(directory)

    for c in cols:

        # define number of buckets
        count_unique = df[c].nunique()
        b = min(count_unique, max_buckets)

        title = ' - '.join([filename.split('.')[0].capitalize(), c.capitalize()])
        if c == 'class':
            df.groupby('class')[c].hist(alpha=1, bins=b)
        else:
            df.groupby('class')[c].plot(kind='kde')
            # df.groupby('class')[c].plot(kind='hist', bins=b, alpha=0.5)
            # df[c].plot(kind='hist')

        plt.title(title)
        plt.legend(classes)
        plt.savefig(os.path.join(directory, c))
        plt.close()

    # does the same as df.groupby('class').hist() but saves all resulting images.
    unique_vals = df['class'].unique()
    for v in unique_vals:
        title = filename.split('.')[0].capitalize() + ' - where Class = ' + str(v)
        df.loc[df['class'] == v].groupby('class').hist()
        plt.savefig(os.path.join(directory, 'histogram_' + str(v)))
        plt.close()
