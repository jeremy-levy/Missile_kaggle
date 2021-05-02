from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier

from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from ML.feature_selection import FeatureSelection
from ML.model import Model, min_acc
from ML.process_data import Data
from utils.help_classes import ScalerEnum, ModelsEnum
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler

from utils.help_classes import ModelsEnum
from sklearn.model_selection import train_test_split
import os
import datetime
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score

import pickle
import numpy as np
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors, svm, metrics

from numpy.linalg import LinAlgError

X = pd.read_csv(os.path.join('data_csv', 'features_train.csv'))
X['time_stamp'] = pd.to_datetime(X['time_stamp'])

X_train = X.loc[X['time_stamp'] < X['time_stamp'].quantile(q=0.9)]
X_val = X.loc[X['time_stamp'] >= X['time_stamp'].quantile(q=0.9)]

y_train = X_train['label']
X_train = X_train.drop(columns=['label', 'time_stamp', 'traj'])

y_val = X_val['label']
X_val = X_val.drop(columns=['label', 'time_stamp', 'traj'])

print('before sampling', X_train.shape)
X_train = X_train.fillna(X_train.mean())

sampling_strategy = "not minority"
rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
X_train, y_train = rus.fit_resample(X_train, y_train)
print('after sampling', X_train.shape)

X_train = X_train.fillna(X_train.median())
X_val = X_val.fillna(X_val.median())

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)

class_weight = {0: 0.3, 1: 10, 2: 2}

level0 = list()
level0.append(('cart', DecisionTreeClassifier(class_weight=class_weight)))
level0.append(('svm', SVC(class_weight=class_weight)))
level0.append(('bayes', GaussianNB()))
level0.append(('rf', RandomForestClassifier(random_state=32, class_weight=class_weight, n_jobs=-1)))
level0.append(('ab', AdaBoostClassifier(algorithm="SAMME", random_state=32)))
# level0.append(('xgb', xgb.XGBClassifier(use_label_encoder=False, random_state=32)))
# level0.append(('cb', CatBoostClassifier(iterations=1000)))
# level0.append(('gb', GradientBoostingClassifier()))

level1 = [('gb_1', StackingClassifier(estimators=level0, final_estimator=GradientBoostingClassifier(), cv=5,
                                      n_jobs=-1)),
          ('cb_1', StackingClassifier(estimators=level0, final_estimator=CatBoostClassifier(iterations=1000), cv=5,
                                      n_jobs=-1)),
          ('xgb_1', StackingClassifier(estimators=level0,
                                       final_estimator=xgb.XGBClassifier(use_label_encoder=False, random_state=32),
                                       cv=5, n_jobs=-1))]

model = StackingClassifier(estimators=level1, final_estimator=LogisticRegression(), cv=5, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

y_true_0 = y_val[y_val == 0]
y_pred_0 = y_pred[y_val == 0]

y_true_1 = y_val[y_val == 1]
y_pred_1 = y_pred[y_val == 1]

y_true_2 = y_val[y_val == 2]
y_pred_2 = y_pred[y_val == 2]

acc_0 = metrics.accuracy_score(y_true_0, y_pred_0)
acc_1 = metrics.accuracy_score(y_true_1, y_pred_1)
acc_2 = metrics.accuracy_score(y_true_2, y_pred_2)

results_dict = {
    "accuracy": [metrics.accuracy_score(y_val, y_pred)],
    'min_acc': [min_acc(y_val, y_pred)],
    'acc_0': [acc_0],
    'acc_1': [acc_1],
    'acc_2': [acc_2],
}
print(results_dict)

X_test = pd.read_csv(os.path.join('data_csv', 'features_test.csv'))

traj = X_test['traj'].values

X_test = X_test.drop(columns=['label', 'time_stamp', 'traj'])
X_test = X_test.fillna(X_test.median())
X_test = scaler.transform(X_test)

final_prediction = model.predict(X_test)

test_predictions = pd.DataFrame({
        'trajectory_ind': traj,
        'label': final_prediction
    })
test_predictions.to_csv('submission_3.csv', index=False)
