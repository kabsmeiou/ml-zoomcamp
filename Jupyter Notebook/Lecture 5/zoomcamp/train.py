#!/usr/bin/env python
# coding: utf-8

# model
from sklearn.linear_model import LogisticRegression
# one hot encoding from sklearn
from sklearn.feature_extraction import DictVectorizer
# for calculating the accuracy
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import auc, roc_auc_score
from sklearn.model_selection import KFold
from tqdm.auto import tqdm
# for splitting the data
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
# ipython
get_ipython().run_line_magic('matplotlib', 'inline')

print('Reading the data...')

# ### Data Preparation
df = pd.read_csv('../../../Datasets/telco/telco.csv');
df.columns = df.columns.str.lower()

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

# convert to numeric because totalcharges is of type object
# ignore errors with errors='coerce'
df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
# fill NaN with zeros
df.totalcharges = df.totalcharges.fillna(0)
# convert target to zeros and ones
df['churn'] = (df['churn'] == 'yes').astype(int)

print('Splitting the dataset...')

# 60, 20, 20 splitting
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

full_train_y = df_full_train.churn.values
train_y = df_train.churn.values
val_y = df_val.churn.values
test_y = df_test.churn.values

del df_train['churn']
del df_val['churn']
del df_test['churn']

numerical_variables = ['tenure', 'monthlycharges', 'totalcharges']
categorical_variables = [column for column in df_full_train.columns if column not in numerical_variables and column != 'churn' and column != 'customerid']

# ### Model
print('Training the model...')

train_dicts = df_train[categorical_variables + numerical_variables].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
# dv.fit_transform(X) fits and immediately transforms the DictVectorizer to a matrix of 0s and 1s
dv.fit(train_dicts)
X_train = dv.transform(train_dicts)
# val
val_dicts = df_val[categorical_variables + numerical_variables].to_dict(orient='records')
X_val = dv.transform(val_dicts)

model = LogisticRegression(max_iter=3000)
model.fit(X_train, train_y)

# predict
y_pred = model.predict_proba(X_val)[:,1]
print(f'Score on Validation Data: {roc_auc_score(val_y, y_pred)}')

def train(data, y, C):
    dicts = data[categorical_variables + numerical_variables].to_dict(orient='records')
    dictv = DictVectorizer(sparse=False)
    X = dictv.fit_transform(dicts)
    
    logit = LogisticRegression(max_iter=4000, C=C)
    logit.fit(X, y)
    return logit, dictv
    
def get_auc(model_train, model_test, model_train_y, model_test_y, C):
    # train model
    logit, fdv = train(model_train, model_train_y, C)
    
    # process testing data
    dicts = model_test[categorical_variables + numerical_variables].to_dict(orient='records')
    X_test = fdv.transform(dicts)
    
    # make predictions
    predictions = logit.predict_proba(X_test)[:,1]
    return roc_auc_score(model_test_y, predictions)

def predict_results(final_model, testing_data, vectorizer):
    # process testing data
    dicts = testing_data[categorical_variables + numerical_variables].to_dict(orient='records')
    X_test = vectorizer.transform(dicts)
    tmp_predictions = final_model.predict_proba(X_test)[:,1]
    return tmp_predictions

print('Validating the model...')
# folds for cross validation
kfold = KFold(n_splits=5, shuffle=True, random_state=1)
counter = 1
fold_scores = []
for train_idx, test_idx in kfold.split(df_full_train):
    # get splits
    model_train = df_full_train.iloc[train_idx]
    model_test = df_full_train.iloc[test_idx]

    # extract churn column
    model_train_y = model_train.churn.values
    model_test_y = model_test.churn.values

    # delete churn from dataset
    del model_train['churn']
    del model_test['churn']

    auc_score = get_auc(model_train, model_test, model_train_y, model_test_y, 1)
    
    # get auc for each fold
    # print(F"AUC {counter}: {auc_score}")
    fold_scores.append(auc_score)
    counter += 1

print('C=%s  %.3f +- %.3f' % (1.0, np.mean(fold_scores), np.std(fold_scores)))

# training the final model
model, fdv = train(df_full_train, full_train_y, 1.0)
predictions = predict_results(model, df_test, fdv)
roc_auc_score(test_y, predictions)
# # process testing data
# dicts = df_test[categorical_variables + numerical_variables].to_dict(orient='records')
# X_test = fdv.transform(dicts)

# # using default solver 
# predictions = model.predict(X_test)
# (predictions == test_y).mean()


# ### Save the model
print('Saving the model...')
# I will be using pickle to import the model
output_file=f'model_C={1.0}.bin'
# open the file
f_out = open(output_file, 'wb')

# use pickle to put our model and dictionary vectorizer inside f_out 
pickle.dump((model, fdv), f_out)

# close the file
f_out.close()

# Another way to do the code above is:
# f_out = open(output_file, 'wb')
with open(output_file, 'wb') as f_out:
    pickle.dump((model, fdv), f_out)
# once outside the 'with' statement, the file will close.



 



