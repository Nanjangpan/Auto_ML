#import libraries

import numpy as np
import pandas as pd
#import time

#import candidate models
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.svm import LinearSVR, LinearSVC, SVC, SVR
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBRegressor, XGBClassifier




from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error

from hyperopt import hp, space_eval, fmin, tpe, hp, STATUS_OK, Trials


def  TPE(X,y,task_type,algo_type):
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
	def split_df(X,y,val_size=0.3):
	    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_size)
	    return X_train, X_test, y_train, y_test
	X_train, X_test, y_train, y_test = split_df(X, y)
	print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

	#Define Parameter Space
	rf_cls_param_space = {
	    'max_depth': hp.choice('max_depth', range(1,30)),
	    'max_features': hp.choice('max_features', range(1,len(X_train.columns))),
	    'min_samples_split': hp.choice('min_samples_split', range(2,100)),
	    'min_samples_leaf': hp.choice('min_samples_leaf', range(1,10)),
	    'n_estimators': hp.choice('n_estimators', range(100,1200)),
	    'criterion': hp.choice('criterion', ["gini", "entropy"])}
	rf_reg_param_space = {
	    'max_depth': hp.choice('max_depth', range(1,30)),
	    'max_features': hp.choice('max_features', range(1,len(X_train.columns))),
	    'min_samples_split': hp.choice('min_sample_split', range(2,100)),
	    'min_samples_leaf': hp.choice('min_sample_leaf', range(1,10)),
	    'n_estimators': hp.choice('n_estimators', range(100,1200)),
	    'criterion': hp.choice('criterion', ["mse", "mae"])}
	svm_param_space = {
    'C': hp.choice('C',[0.001, 0.01, 0.1, 1, 10, 100, 1000])}
	sgd_reg_param_space = {
    	'loss' : hp.choice('loss',["squared_loss","huber","epsilon_insensitive","squared_epsilon_insensitive"]),
    	'n_iter': hp.choice('n_iter',[0.001, 0.01, 0.1, 1, 10, 100, 1000]),  
    	'alpha': hp.choice('alpha',[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]),   
    	'penalty': hp.choice('penalty',['none', 'l2', 'l1', 'elasticnet'])}
	sgd_cls_param_space = {
    	'loss' : hp.choice('loss',["hinge","log","modified_huber","squared_hinge","perceptron"]),
    	'alpha': hp.choice('alpha',[0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000]),    #need to check
    	'penalty': hp.choice('penalty',['none', 'l2', 'l1', 'elasticnet'])}
	MultinomialNB_param_space = {
		'alpha':hp.choice('alpha',[0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000]),
		'fit_prior':hp.choice('fit_prior',[True,False])}
	KNeighborsClassifier_param_space = {
		'n_neighbors': hp.choice('n_neighbors',range(1,30)),
		'weights' : hp.choice('weights',["uniform","distance"])
	}
	GradientBoostingClassifier_param_space = {
		'loss':  hp.choice('loss',["deviance","exponential"]),
		'learning_rate': hp.choice('learning_rate',[0.15,0.1,0.05,0.01,0.005,0.001]),
		'n_estimators': hp.choice('n_estimators',[100,250,500,750,1000,1250,1500,1750]),
	    'min_samples_split': hp.choice('min_samples_split', range(2,100)),
	    'min_samples_leaf': hp.choice('min_samples_leaf', range(1,10)),
	    'max_depth': hp.choice('max_depth', range(1,30))
	}
	GradientBoostingRegressor_param_space = {
		'loss':  hp.choice('loss',["ls","lad","huber","quantile"]),
		'learning_rate': hp.choice('learning_rate',[0.15,0.1,0.05,0.01,0.005,0.001]),
		'n_estimators': hp.choice('n_estimators',[100,250,500,750,1000,1250,1500,1750]),
	    'min_samples_split': hp.choice('min_samples_split', range(2,100)),
	    'min_samples_leaf': hp.choice('min_samples_leaf', range(1,10)),
	    'max_depth': hp.choice('max_depth', range(1,30))
	}
	DecisionTreeClassifier_param_space = {
		'criterion': hp.choice('criterion',["gini","entropy"]),
	    'min_samples_split': hp.choice('min_samples_split', range(2,100)),
	    'min_samples_leaf': hp.choice('min_samples_leaf', range(1,10))
	}
	XGB_param_space = {
		'learning_rate ': hp.choice('learning_rate',[0.01,0.015,0.025,0.05,0.1,0.3]) ,
		'gamma ': hp.choice('gamma',[0.05,0.1,0.3,0.5,0.7,0.9,1]),
		'max_depth ':  hp.choice('max_depth',[2,4,6,9,12,15,17,25]),
		'': hp.choice('',[]),
		'subsample ': hp.choice('subsample',[0.6,0.7,0.8,0.9,1]),
		'': hp.choice('',[]),
		'reg_lambda': hp.choice('reg_lambda',[0,0.01,0.1,1]),
		'reg_alpha': hp.choice('reg_alpha',[0,0.1,0.5,1]), 
		'n_jobs': hp.choice('n_jobs',[-1])

	}
	param_space = {'RandomForestClassifier':rf_cls_param_space,
	               'RandomForestRegressor':rf_reg_param_space,
	               'LinearSVC': svm_param_space,
	               'LinearSVR': svm_param_space,
	               'SVC':svm_param_space,
	               'SVR':svm_param_space,
	               'SGDRegressor': sgd_reg_param_space,
	               'SGDClassifier':sgd_cls_param_space,
	               'MultinomialNB': MultinomialNB_param_space,
	               'KNeighborsClassifier': KNeighborsClassifier_param_space,
	               'GradientBoostingClassifier': GradientBoostingClassifier_param_space ,
	               'GradientBoostingRegressor': GradientBoostingRegressor_param_space,
	               'DecisionTreeClassifier': DecisionTreeClassifier_param_space,
	               'XGBRegressor': XGB_param_space,
	               'XGBClassifier': XGB_param_space
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
	    best = fmin(f_reg, param_space[str(algo_type).split('(')[0]], algo=tpe.suggest, max_evals=5, trials=trials)
	elif task_type == 'cls':
	    best = fmin(f_cls, param_space[str(algo_type).split('(')[0]], algo=tpe.suggest, max_evals=5, trials=trials)
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
	df = pd.read_csv("../Dataset/dataset_with_perday.csv")
	X = df[list(df.columns)[:-1]]
	X.replace(np.inf,0,inplace=True)
	y = df[list(df.columns)[-1]]
	algo_type = RandomForestClassifier() #Will be given
	task_type = 'cls'                   #Will be given
	finalModel = TPE(X,y,task_type,algo_type)
	print(str(finalModel.get_params))

	#LinearSVR, LinearSVC, SVC, SVR
# 	from sklearn.linear_model import SGDRegressor, SGDClassifier
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
# from sklearn.tree import DecisionTreeClassifier
# from xgboost import XGBRegressor, XGBClassifier
	# df = pd.read_csv("../Dataset/heart_target.csv")
	# X = df[list(df.columns)[:-1]]
	# X.replace(np.inf,0,inplace=True)
	# y = df[list(df.columns)[-1]]
	# algo_type = DecisionTreeClassifier() #Will be given
	# task_type = 'cls'                   #Will be given
	# finalModel = TPE(X,y,task_type,algo_type)
	# print(str(finalModel.get_params))