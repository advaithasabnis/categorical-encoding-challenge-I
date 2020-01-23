# -*- coding: utf-8 -*-
"""
@author: Advait Hasabnis
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#%% Importing data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
alldata = [train, test]

#%% Cleaning data

# Replacing infrequent values
vc_nom_6 = train.nom_6.value_counts()
vc_nom_7 = train.nom_7.value_counts()
vc_nom_8 = train.nom_8.value_counts()
vc_nom_9 = train.nom_9.value_counts()

infrequent_nom_6 = list(vc_nom_6[vc_nom_6 < 20].index)
infrequent_nom_7 = list(vc_nom_7[vc_nom_7 < 7].index)
infrequent_nom_8 = list(vc_nom_8[vc_nom_8 < 5].index)
infrequent_nom_9 = list(vc_nom_9[vc_nom_9 < 3].index)

for dataset in alldata:
    dataset.loc[dataset.nom_6.isin(infrequent_nom_6), 'nom_6'] = 'fffffffff'
    dataset.loc[dataset.nom_7.isin(infrequent_nom_7), 'nom_7'] = 'fffffffff'
    dataset.loc[dataset.nom_8.isin(infrequent_nom_8), 'nom_8'] = 'fffffffff'
    dataset.loc[dataset.nom_9.isin(infrequent_nom_9), 'nom_9'] = 'fffffffff'

# Replacing values that exist in test but not in train
test.loc[~test.nom_8.isin(train.nom_8.unique()), 'nom_8'] = 'fffffffff'
test.loc[~test.nom_9.isin(train.nom_9.unique()), 'nom_9'] = 'fffffffff'

# Replacing values that exist in train but not in test
train.loc[~train.nom_7.isin(test.nom_7.unique()), 'nom_7'] = 'fffffffff'
train.loc[~train.nom_8.isin(test.nom_8.unique()), 'nom_8'] = 'fffffffff'
train.loc[~train.nom_9.isin(test.nom_9.unique()), 'nom_9'] = 'fffffffff'

#%% Preprocessing pipeline
bin_cols = ['bin_1', 'bin_2', 'bin_3', 'bin_4']
nom_gen_cols = ['day', 'month', 'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7']
nom_spe_cols = ['nom_8', 'nom_9']
ord_gen_cols = ['ord_0', 'ord_3', 'ord_4', 'ord_5']
ord_spe_cols = ['ord_1', 'ord_2']

bin_transformer = OrdinalEncoder()

nom_gen_transformer = OneHotEncoder(categories='auto', drop='first')

nom_categories = []
for col in nom_spe_cols:
    nom_categories.append(np.append(train[col].unique(), 'fffffffff'))

nom_spe_transformer = OneHotEncoder(categories=nom_categories, drop='first')

ord_gen_transformer = Pipeline(steps=[
        ('autoord2', OrdinalEncoder()),
        ('minmax4', MinMaxScaler())
        ])

ord_spe_transformer = Pipeline(steps=[
        ('speord1', OrdinalEncoder(categories=[['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster'],
                                              ['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot']
                                              ])),
        ('minmax5', MinMaxScaler())
        ])
    
preprocessor = ColumnTransformer(transformers=[
        ('bin', bin_transformer, bin_cols),
        ('nom_gen', nom_gen_transformer, nom_gen_cols),
        ('nom_spe', nom_spe_transformer, nom_spe_cols),
        ('ord_gen', ord_gen_transformer, ord_gen_cols),
        ('ord_spe', ord_spe_transformer, ord_spe_cols)
        ])

#%% Train and Test set
X_train = train.drop(['target'], axis=1).copy()
y_train = train.target.copy()
X_test = test.copy()

#%% Preprocessing data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

#%% Logistic regression
log_regressor = LogisticRegression(solver='saga', C=0.12, max_iter=400)
log_regressor.fit(X_train, y_train)
y_pred_log = log_regressor.predict_proba(X_test)[:, 1]
log_coef = log_regressor.coef_

# Cross validation
log_score = cross_val_score(log_regressor, X_train, y_train, scoring='roc_auc', cv=5, n_jobs=-1, verbose=1).mean()
print(log_score)

#%% Saving prediction file
dataPred = pd.DataFrame()
dataPred['id'] = test['id'].copy()
dataPred['target'] = y_pred_log
dataPred.to_csv('pb_03.csv', index=False, header=True)

#%% Parameter tuning for logistic regression
all_grid = {'C': [0.1, 0.12, 0.14],
            'solver': ['saga'],
            'fit_intercept': [True],
            'max_iter': [400],
            'penalty': ['l2'],
            'class_weight': [None]
            }
log_grid = GridSearchCV(estimator=LogisticRegression(), param_grid=all_grid, cv=5, n_jobs=-1, scoring='roc_auc')
log_grid.fit(X_train, y_train)
log_grid.best_params_

