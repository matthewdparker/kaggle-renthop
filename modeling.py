"""
Script creates, trains, and executes XGBoost model against train and test data, and exports resulting predictions for test data to .csv file at location the user specifies at runtime.

Execute the following in the terminal to run:
python modeling.py train_munged_path test_munged_path save_predictions_path

Script constructed with some starter code from SRK: https://www.kaggle.com/sudalairajkumar/two-sigma-connect-rental-listing-inquiries/xgb-starter-in-python
"""

import numpy as np
import pandas as pd
import scipy.stats as scs
import matplotlib.dates as mdates

# model imports
import xgboost as xgb

# imports from starter code by SRK
import os
import sys
import operator
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def model(munged_train_filepath, munged_test_filepath, save_predictions_path):
    train_df = pd.read_json(munged_train_filepath)
    test_df = pd.read_json(munged_test_filepath)


    # Need to ask John what this does
    train_df['features'] = train_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
    test_df['features'] = test_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
    tfidf = CountVectorizer(stop_words='english', max_features=200)
    tr_sparse = tfidf.fit_transform(train_df["features"])
    te_sparse = tfidf.transform(test_df["features"])


    # which columns are currently numeric?
    numeric = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_idx = [column for column in train_df.columns if train_df[column].dtype in numeric]
    non_numeric_idx = [column for column in train_df.columns if column not in numeric_idx]
    train_df[numeric_idx].head()


    # seaparate train and test into X and y
    train_X = sparse.hstack([train_df[numeric_idx], tr_sparse]).tocsr()
    test_X = sparse.hstack([test_df[numeric_idx], te_sparse]).tocsr()

    target_num_map = {'high':0, 'medium':1, 'low':2}
    train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))


    # function to create and run model
    def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=1000):
        param = {}
        param['objective'] = 'multi:softprob'
        param['eta'] = 0.1
        param['max_depth'] = 6
        param['silent'] = 1
        param['num_class'] = 3
        param['eval_metric'] = "mlogloss"
        param['min_child_weight'] = 1
        param['subsample'] = 0.7
        param['colsample_bytree'] = 0.7
        param['nthread'] = 4
        param['seed'] = seed_val
        num_rounds = num_rounds

        plst = list(param.items())
        xgtrain = xgb.DMatrix(train_X, label=train_y)

        if test_y is not None:
            xgtest = xgb.DMatrix(test_X, label=test_y)
            watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
            model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)
        else:
            xgtest = xgb.DMatrix(test_X)
            model = xgb.train(plst, xgtrain, num_rounds)

        pred_test_y = model.predict(xgtest)
        return pred_test_y, model


    # Run model and export to specified filepath
    preds, model = runXGB(train_X, train_y, test_X, num_rounds=300)
    out_df = pd.DataFrame(preds)
    out_df.columns = ["high", "medium", "low"]
    out_df["listing_id"] = test_df.listing_id.values
    out_df.to_csv(save_predictions_path, index=False)


if __name__ == '__main__':
    # import munged data
    munged_train_filepath = sys.argv[1]
    munged_test_filepath = sys.argv[2]
    save_predictions_path = sys.argv[3]

    # model munged data
    model(munged_train_filepath, munged_test_filepath, save_predictions_path)
