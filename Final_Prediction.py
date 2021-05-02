import os
import pickle

import joblib
import pandas as pd
import numpy as np

# from Classification import log_features
from ML.model import min_acc


def make_final_prediction(filename_test, filename_train, name_model, features_selected):
    text_file = open(features_selected, "r")
    features_selected = text_file.readlines()
    features_selected = [x.replace('\n', '') for x in features_selected]
    text_file.close()

    loaded_model = pickle.load(open(name_model, 'rb'))

    X = pd.read_csv(filename_test)
    traj = X['traj'].values

    X = X.drop(columns=['label', 'time_stamp', 'traj'])
    X = X[features_selected]

    # data_train = pd.read_csv(filename_train)
    # data_train = data_train[features_selected]

    X = X.fillna(X.mean())
    # for feat in log_features:
    #     X[feat] = np.log(X[feat] - X[feat].min() + 0.05)

    scaler = joblib.load('scaler.pkl')
    X = scaler.transform(X)
    prediction = loaded_model.predict(X)

    test_predictions = pd.DataFrame({
        'trajectory_ind': traj,
        'label': prediction
    })
    test_predictions.to_csv('submission_1.csv', index=False)


# X = pd.read_csv(os.path.join('data_csv', 'features_train.csv'))
#
# X['time_stamp'] = pd.to_datetime(X['time_stamp'])
# X_train = X.loc[X['time_stamp'] < X['time_stamp'].quantile(q=0.8)]
# X_val = X.loc[X['time_stamp'] >= X['time_stamp'].quantile(q=0.8)]
# X_val.to_csv(os.path.join('data_csv', 'features_val.csv'))

make_final_prediction(filename_test=os.path.join('data_csv', 'features_test.csv'), name_model='ModelsEnum.RF',
                      features_selected='list_features.txt',
                      filename_train=os.path.join('data_csv', 'features_train.csv'))

# make_final_prediction(filename_test=os.path.join('data_csv', 'features_val.csv'), name_model='ModelsEnum.RF',
#                       features_selected='list_features.txt',
#                       filename_train=os.path.join('data_csv', 'features_train.csv'))

# submission_test = pd.read_csv('submission_1.csv')
# print(min_acc(X_val['label'].values, submission_test['label'].values))
