#import libraries

import numpy as np
import pandas as pd
#import time

#import candidate models
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.forest import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error

from hyperopt import hp, space_eval, fmin, tpe, hp, STATUS_OK, Trials


def  TPE(df,target,task_type,algo_type):
	'''
	Parameters:
	df: pandas.DataFrame() Data Including both Xs and y
	target: str Name of the target(Y) column
	task_type: str ('cls' or 'reg')
	algo_type: classification or regression model object
	--------------------------------------------------------
	For this to work we need:
	1. The algorithm object having .set_params method
	2. str(algo_type).split('(')[0] Must return the name of the ObjectType
	'''
	#import Data and split into X,y - train,test
	def split_df(df,target,val_size=0.3):
	    X_lst = list(df.columns)
	    X_lst.remove(target)
	    X = df[X_lst]
	    y = df[target]
	    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_size)
	    return X_train, X_test, y_train, y_test
	X_train, X_test, y_train, y_test = split_df(df, target)
	print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

	#Define Parameter Space
	rf_cls_param_space = {
	    'max_depth': hp.choice('max_depth', range(1,20)),
	    'max_features': hp.choice('max_features', range(1,len(X_train.columns))),
	    'n_estimators': hp.choice('n_estimators', range(100,500)),
	    'criterion': hp.choice('criterion', ["gini", "entropy"])}
	rf_reg_param_space = {
	    'max_depth': hp.choice('max_depth', range(1,20)),
	    'max_features': hp.choice('max_features', range(1,len(X_train.columns))),
	    'n_estimators': hp.choice('n_estimators', range(100,500)),
	    'criterion': hp.choice('criterion', ["mse", "mae"])}

	param_space = {'RandomForestClassifier':rf_cls_param_space,
	               'RandomForestRegressor':rf_reg_param_space,
	               
	              }

	#Define score functions
	def score_model_classification(model,params):
	    return -1*cross_val_score(model.set_params(**params),X_train,y_train,cv=5,scoring='f1').mean()
	def score_model_regression(model,params):
	    ### cross_val_score()reuturns negative mse
	    return -1 - cross_val_score(model.set_params(**params),X_train,y_train,cv=5,scoring='neg_mean_squared_error').mean()
	def f_cls(params):
	    score = score_model_classification(algo_type,params)
	    print('score config (-f1):', score, params)
	    return {'loss': score, 'status': STATUS_OK}
	def f_reg(params):
	    score = score_model_regression(algo_type,params)
	    print('score config (-1 + mse):', score, params)
	    return {'loss': score, 'status': STATUS_OK}

	#Optimize Hyperparameters
	trials = Trials()
	if task_type == 'reg':
	    best = fmin(f_reg, param_space[str(algo_type).split('(')[0]], algo=tpe.suggest, max_evals=100, trials=trials)
	elif task_type == 'cls':
	    best = fmin(f_cls, param_space[str(algo_type).split('(')[0]], algo=tpe.suggest, max_evals=100, trials=trials)
	print ('best:')
	print (best)


	#Train Model with optimized Hyperparameters
	params = space_eval(param_space[str(algo_type).split('(')[0]],best)
	algo_type.set_params(**params)
	algo_type.fit(X_train,y_train)

	#Final Evaluation
	def test_classification(model,params):
	    y_pred = model.predict(X_test)
	    return y_pred, f1_score(y_test, y_pred)
	def test_regression(model,params):
	    y_pred = model.predict(X_test)
	    return y_pred, mean_squared_error(y_test, y_pred)

	y_pred = None
	if task_type == 'cls':
	    y_pred, f1 = test_classification(algo_type, best)
	    print(f1)
	else:
	    y_pred, mse = test_regression(algo_type,best)
	    print(mse)
	return algo_type


if __name__ == '__main__':
	df = pd.read_csv("../Dataset/heart_target.csv") #Will be given
	algo_type = RandomForestClassifier() #Will be given
	task_type = 'cls'                   #Will be given
	finalModel = TPE(df,"target",task_type,algo_type)
	print(str(finalModel.get_params))
