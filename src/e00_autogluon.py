"""
    This file runs lightgbm, train with train data and predict test data.
"""

import pandas as pd
import numpy as np
import gc
import autogluon as ag
import joblib
from sklearn.model_selection import StratifiedKFold
from util_logger import get_logger

import sys
#argvs = sys.argv 
#_ , runtype, version = argvs
LOG = get_logger()
LOG.info("start e00")

def run_autogluon(labels, weights, data):

    # convert data into ag.TabularPrediction.Dataset. 
    # train using 80% of the data, 20% of the data is used for watchlist
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state = 123)
    idx_train, idx_test = list(skf.split(data, labels))[0]

    passed_df = pd.DataFrame(data)
    passed_df['label'] = labels
    dtrain = ag.TabularPrediction.Dataset(df=passed_df.iloc[idx_train,:])
    dvalid = ag.TabularPrediction.Dataset(df=passed_df.iloc[idx_test,:])

    # set parameters
    label_column = 'label'

    # run autogluon
    LOG.info("started autogluon")
    model = ag.TabularPrediction.fit(train_data=dtrain, problem_type='regression', label=label_column, tuning_data=dvalid, output_directory='../model/autogluon.model')

    return model

def predict_and_write(in_fname, out_fname):
    autogluondata = pd.DataFrame(joblib.load(in_fname)['data'])

    # to reduce memory consumption, predict by small chunks.
    def predict(data):
        return pd.Series(model.predict(data)) 

    data_list = np.array_split(autogluondata, 8)
    preds = pd.concat([predict(data) for data in data_list])
    preds.to_csv(out_fname)

# train model
autogluondata_train = joblib.load( '../model/autogluondata_train.pkl') 
model = run_autogluon(autogluondata_train["labels"], autogluondata_train["weights"], autogluondata_train["data"])

# release memory
autogluondata_train = None
gc.collect()

# predict test data and write
predict_and_write( '../model/autogluondata_test.pkl', "../model/autogluon_predicted.txt") 

LOG.info("finished")

