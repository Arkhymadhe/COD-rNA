import os
import pandas as pd

import zlib
import zipfile

from sklearn.dataset import load_svmlight_file
from sklearn.model_selection import train_test_split


def load_dataset(path):
    ''' Load svmlight dataset from path. '''
    
    feature_file, target_file = load_svmlight_file(path)
    
    return pd.DataFrame(file.todense()), target_file


def manual_describe(data, path, save = False):
    ''' Return a DataFrame containing a description of missing values, unique values, and data types. '''
    
    desc = {'Name' : pd.Series(np.array(data.columns) + 1).apply(lambda x: f'Feature {x}'),
            'Null' : data.isnull().sum(),
            'Null (%)' : 100 * data.isnull().sum()/data.shape[0],
            'Unique' : data.nunique(),
            'Unique (%)' : 100 * data.nunique()/data.shape[0],
            'Dtypes' : data.dtypes}

    desc = pd.DataFrame(desc)
    
    if save:
        desc.to_csv(os.path.join(path, 'manual_description.csv'), index = False)
    
    return desc


def auto_description(data):
    ''' Return auto-generated decsription of dataset. '''
    
    return data.describe()


def split_data(X, y, split_size = 0.25, random_state = 42):
    ''' Split dataset into folds. '''
    
    X, X_, y, y_ = train_test_split(X, y, test_size = split_size, stratify = y, random_state = random_state)
    
    return X, X_, y, y_


