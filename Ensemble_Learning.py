import os

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import make_scorer
from sklearn.preprocessing import MinMaxScaler

from ML.model import Model, min_acc
from utils.help_classes import ModelsEnum

###  Split data into 3 datasets  ###

X = pd.read_csv(os.path.join('data_csv', 'features_train.csv'))
X['time_stamp'] = pd.to_datetime(X['time_stamp'])

X_train_1 = X.loc[X['time_stamp'] < X['time_stamp'].quantile(q=0.7)]

X_others = X.loc[X['time_stamp'] >= X['time_stamp'].quantile(q=0.7)]
X_train_2 = X_others.loc[X_others['time_stamp'] < X_others['time_stamp'].quantile(q=0.5)]
X_val = X_others.loc[X_others['time_stamp'] >= X_others['time_stamp'].quantile(q=0.5)]

print('X_train_1', X_train_1.shape)
print('X_train_2', X_train_2.shape)
print('X_val', X_val.shape)

assert X_train_1.shape[0] + X_train_2.shape[0] + X_val.shape[0] == 32741

### Train first layer ###

y_train_1 = X_train_1['label']
X_train_1 = X_train_1.drop(columns=['label', 'time_stamp', 'traj'])

y_val = X_val['label']
X_val = X_val.drop(columns=['label', 'time_stamp', 'traj'])

print('before sampling', X_train_1.shape)
X_train_1 = X_train_1.fillna(X_train_1.mean())
X_val = X_val.fillna(X_train_1.mean())

sampling_strategy = "not minority"
rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
X_train_1, y_train_1 = rus.fit_resample(X_train_1, y_train_1)
print('after sampling', X_train_1.shape)

scaler = MinMaxScaler()
scaler.fit(X_train_1)
X_train_1 = scaler.transform(X_train_1)
X_val = scaler.transform(X_val)

all_classifiers = [ModelsEnum.Xgboost, ModelsEnum.RF, ModelsEnum.AB]
its = [20, 200, 20]
trained_classifier = []
class_weight = {0: 0.3, 1: 5, 2: 2}

for it, cl in zip(its, all_classifiers):
    ml_model = Model(model_name="time_series_3_" + str(cl), classifier_method=cl,  verbose=True, binary_task=False,
                     write_csv=True, class_weight=class_weight)
    trained_model, results_train = ml_model.cross_fold_validation(X_train_1, y_train_1, grid_search=False, cv=3,
                                                                  n_iter=it, scoring=make_scorer(min_acc), n_jobs=30)
    trained_classifier.append(trained_model)
    results_dict = ml_model.test_set(X_val, y_val)


### 2nd layer ###

y_train_2 = X_train_2['label']
X_train_2 = X_train_2.drop(columns=['label', 'time_stamp', 'traj'])
X_train_2 = X_train_2.fillna(X_train_2.median())
X_train_2 = scaler.transform(X_train_2)

print('before sampling', X_train_2.shape)

sampling_strategy = "not minority"
rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
X_train_2, y_train_2 = rus.fit_resample(X_train_2, y_train_2)
print('after sampling', X_train_2.shape)

predictions_train_2 = np.empty(shape=(X_train_2.shape[0], 0))
predictions_val = np.empty(shape=(X_val.shape[0], 0))

for model in trained_classifier:
    model_prediction_train_2 = model.predict_proba(X_train_2)
    predictions_train_2 = np.concatenate((predictions_train_2, model_prediction_train_2), axis=1)

    model_prediction_test = model.predict_proba(X_val)
    predictions_val = np.concatenate((predictions_val, model_prediction_test), axis=1)

print("predictions_train_2", predictions_train_2.shape)
print("predictions_val", predictions_val.shape)

ml_model = Model(model_name="time_series_3_" + str(ModelsEnum.LR), classifier_method=ModelsEnum.LR, verbose=True,
                 binary_task=False, write_csv=True)

trained_model, results_train = ml_model.cross_fold_validation(predictions_train_2, y_train_2, grid_search=False, cv=3,
                                                              n_iter=300, scoring=make_scorer(min_acc), n_jobs=2)
results_dict = ml_model.test_set(predictions_val, y_val)


X_test = pd.read_csv(os.path.join('data_csv', 'features_test.csv'))
traj = X_test['traj'].values
X_test = X_test.drop(columns=['label', 'time_stamp', 'traj'])
X_test = X_test.fillna(X_test.median())
X_test = scaler.transform(X_test)

predictions_test = np.empty(shape=(X_test.shape[0], 0))

for i, model in enumerate(trained_classifier):
    if i == 1:
        continue
    model_prediction_test = model.predict_proba(X_test)
    predictions_test = np.concatenate((predictions_test, model_prediction_test), axis=1)

final_prediction = trained_model.predict(predictions_test)
test_predictions = pd.DataFrame({
        'trajectory_ind': traj,
        'label': final_prediction
    })
test_predictions.to_csv('submission_3_EnsembleLearning.csv', index=False)