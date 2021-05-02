import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import joblib

from Feature_Selection import feature_selection
from ML.model import Model, min_acc
from utils.help_classes import ModelsEnum
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)

X = pd.read_csv(os.path.join('data_csv', 'features_train.csv'))

X['time_stamp'] = pd.to_datetime(X['time_stamp'])
X_train = X.loc[X['time_stamp'] < X['time_stamp'].quantile(q=0.9)]
X_val = X.loc[X['time_stamp'] >= X['time_stamp'].quantile(q=0.9)]

y_train = X_train['label']
y_val = X_val['label']

data_train_0 = X_train.drop_duplicates(subset=['traj']).loc[X_train['label'] == 0]
weight_0 = (X_train.shape[0] - data_train_0.shape[0]) / X_train.shape[0]            # 0.31
data_train_1 = X_train.drop_duplicates(subset=['traj']).loc[X_train['label'] == 1]
weight_1 = (X_train.shape[0] - data_train_1.shape[0]) / X_train.shape[0]            # 0.76
data_train_2 = X_train.drop_duplicates(subset=['traj']).loc[X_train['label'] == 2]
weight_2 = (X_train.shape[0] - data_train_2.shape[0]) / X_train.shape[0]            # 0.93

X_train = X_train.drop(columns=['label', 'time_stamp', 'traj'])
X_val = X_val.drop(columns=['label', 'time_stamp', 'traj'])

X_train = X_train.fillna(X_train.mean())
X_val = X_val.fillna(X_val.mean())

print('before sampling', X_train.shape)
sampling_strategy = "not minority"
rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
# rus = RandomOverSampler(sampling_strategy=sampling_strategy)
X_train, y_train = rus.fit_resample(X_train, y_train)
print('after sampling', X_train.shape)

class_weight = {0: weight_0, 1: weight_1, 2: weight_2}
# class_weight = {0: 1, 1: 1, 2: 1}

feature_names = X_train.columns
with open('list_features_2.txt', 'w') as filehandle:
    for listitem in feature_names:
        filehandle.write('%s\n' % listitem)

# log_features = ['coefs_0', 'higher_z', 'mean_acc', 'q75_acc', 'median_acc', 'energy_z', 'min_fft', 'down_fft']
# for feat in log_features:
#     X_train[feat] = np.log(X_train[feat] - X_train[feat].min() + 0.05)
#     X_val[feat] = np.log(X_val[feat] - X_val[feat].min() + 0.05)

classifier_method = ModelsEnum.Xgboost
print('for model', classifier_method, 'results are ')
print('X_train', X_train.shape)
print('X_val', X_val.shape)

ml_model = Model(model_name="time_series_2_" + str(classifier_method),
                 classifier_method=classifier_method, verbose=True,
                 binary_task=False, write_csv=True,
                 class_weight=class_weight)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
joblib.dump(scaler, 'scaler_2.pkl')

trained_model, results_train = ml_model.cross_fold_validation(X_train, y_train, grid_search=False, cv=3,
                                                              n_iter=100, scoring=make_scorer(min_acc), n_jobs=40)

pickle.dump(trained_model, open(str(classifier_method), 'wb'))
results_dict = ml_model.test_set(X_val, y_val)

# best_params_ = trained_model.best_params_
# RF_model = RandomForestClassifier(**best_params_)
# RF_model.fit(X_train_class, y_train)
#
# importance = RF_model.feature_importances_
# indices = np.argsort(importance)
# scores = importance[indices]
# list_features = np.array(feature_names)
# features = list_features[indices]
#
# features = features[-30:]
# scores = scores[-30:]
#
# plt.figure()
# plt.barh(features, scores)
# plt.savefig(fname='feature_importance')





