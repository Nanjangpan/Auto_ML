# scikit-learn algorithm cheat-sheet
import time
import csv
import numpy as np
import pandas as pd
#import libraries


#import candidate models
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRegressor, XGBClassifier




from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error

from hyperopt import hp, space_eval, fmin, tpe, hp, STATUS_OK, Trials

#openML
#--------------------------------------------------------------------------------------------
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
   lin_svc_param_space = {
    #'kernel': hp.choice('kernel',['poly', 'rbf', 'sigmoid']), 
    #'gamma': hp.choice('gamma',[1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5, 0.6, 0.9]),
    'C': hp.choice('C',[1, 10, 100, 1000, 10000])}

   lin_svr_param_space = {
    #'kernel': hp.choice('kernel',['poly', 'rbf', 'sigmoid']), 
    #'gamma': hp.choice('gamma',[1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5, 0.6, 0.9]),
    'C': hp.choice('C',[1, 10, 100, 1000, 10000])}



   param_space = {'RandomForestClassifier':rf_cls_param_space,
                  'RandomForestRegressor':rf_reg_param_space,
                  'LinearSVC': lin_svc_param_space,
                  'LinearSVR': lin_svr_param_space
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
   print (최고)


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

'''
#Mice
#--------------------------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import Imputer


class MiceImputer:
  
    model_dict_ = {}
    
    def __init__(self, seed_nulls=False, seed_strategy='mean'):
        self.seed_nulls = seed_nulls
        self.seed_strategy = seed_strategy
        
    
    def transform(self, X):
        col_order = X.columns
        new_X = []
        mutate_cols = list(self.model_dict_.keys())
        
        for i in mutate_cols:
            y = X[i]
            x_null = X[y.isnull()]
            y_null = y[y.isnull()].reset_index()['index']
            y_notnull = y[y.notnull()]
            
            model = self.model_dict_.get(i)
            
            if self.seed_nulls:
                x_null = model[1].transform(x_null)
            else:
                null_check = x_null.isnull().any()
                x_null = x_null[null_check.index[~null_check]]
            
            pred = pd.concat([pd.Series(model[0].predict(x_null))\
                              .to_frame()\
                              .set_index(y_null),y_notnull], axis=0)\
                              .rename(columns={0: i})
            
            new_X.append(pred)

        new_X.append(X[X.columns.difference(mutate_cols)])

        final = pd.concat(new_X, axis=1)[col_order]

        return final
        
        
    def fit(self, X):      
        x = X.fillna(value=np.nan)

        null_check = x.isnull().any()
        null_data = x[null_check.index[null_check]]
        
        for i in null_data:
            y = null_data[i]
            y_notnull = y[y.notnull()]

            model_list = []
            if self.seed_nulls:
                imp = Imputer(strategy=self.seed_strategy)
                model_list.append(imp.fit(x))
                non_null_data = pd.DataFrame(imp.fit_transform(x))
                
            else:
                non_null_data = x[null_check.index[~null_check]]
                
            
            x_notnull = non_null_data[y.notnull()]
            
            if y.nunique() > 2:
                model = LinearRegression()
                model.fit(x_notnull, y_notnull)
                model_list.insert(0, model)
                self.model_dict_.update({i: model_list})
            else:
                model = LogisticRegression()
                model.fit(x_notnull, y_notnull)
                model_list.insert(0, model)
                self.model_dict_.update({i: model_list})

        return self
        

    def fit_transform(self, X):
        return self.fit(X).transform(X)
'''
#--------------------------------------------------------------------------------------------
dataset = pd.read_csv('./data/housePrice_SalePrice.csv', engine='python')
data_size = dataset.shape[0]
a = dataset.iloc[:, -1][1]
if(issubclass(type(a), str)): #check predict data is stirng
  str_type = True
else :
  str_type = False

choose = ''

#if No, we use this function
def No():
  global choose
  choose = "N"
  print("No\n")
  time.sleep(1)

#if Yes, we use this function    
def Yes():
  global choose
  choose = "Y"
  print("Yes\n")
  time.sleep(1)

'''
#oversampling(<10%)
def Oversampling():
  from sklearn.datasets import make_classification
  from sklearn.decomposition import PCA
  from imblearn.over_sampling import SMOTE
  # 모델설정
  global x_train, y_train
  sm = SMOTE(ratio='auto', kind='regular')
  x_train, y_train = sm.fit_sample(x_train,list(y_train))

onehot_col = []
#onehotencoding
def OnehotEncoding():
  global dataset
  global exclude_col 
  dataset = pd.get_dummies(dataset, columns=onehot_col)

def scaling():
  from sklearn.preprocessing import StandardScaler
  global x_train
  scaler = StandardScaler()
  x_train = scaler.fit_transform(x_train)


#missing value(not working with string, so we first use onehotencoding)
def fill():
  MICE = MiceImputer()
  global dataset
  dataset = MICE.fit(dataset).transform(dataset)
'''
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

dataset = pd.get_dummies(dataset)
#MICE = MiceImputer()
#dataset = MICE.fit(dataset).transform(dataset)
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)



print("Start")
time.sleep(1)

print(">50 samples") #check data size
if data_size > 50 : 
    Yes()
else :
    No()

if choose == "N":
    print("----------------------------------------------------------")
    print("get more data")
    print("----------------------------------------------------------")
elif choose == "Y":
    print("")
    print("predicting a category")
    choose = input("Enter Y or N ")
    if choose == "N":
        print("")
        print("predicting a quantity")
        choose = input("Enter Y or N ")
        if choose == "N":
            print("")
            print("just looking")
            choose = input("Enter Y or N ")
            if choose =="N":
                print("----------------------------------------------------------")
                print("predicitng structure")
                time.sleep(1)
                print("tough luck")
                print("----------------------------------------------------------")
            elif choose == "Y":
                print("----------------------------------------------------------")
                print("Randomized PCA")
                print("----------------------------------------------------------")
                print("")
                choose = input("If it doesn't work, press Y. ")
                if choose =="Y":
                    print("")
                    print("<10K samples")
                    choose = input("Enter Y or N ")
                    if choose == "N":
                        print("----------------------------------------------------------")
                        print("kernel approximation")
                        print("----------------------------------------------------------")
                    elif choose == "Y":
                        print("----------------------------------------------------------")
                        print("Isomap")
                        print("Spectral Embedding")
                        print("----------------------------------------------------------")
                        print("")
                        choose = input("If it doesn't work, press Y. ")
                        if choose == "Y":
                            print("----------------------------------------------------------")
                            print("LLE")
                            print("----------------------------------------------------------")
        elif choose == "Y":
            print("")
            print("<100K samples")
            if data_size < 100000 : 
                Yes()
            else :
                No()
            if choose == "N":
                print("----------------------------------------------------------")
                print("SGD Regressor")
                print("----------------------------------------------------------")
                algo_type = SGDRegressor() 
                task_type = 'reg'                  
                sgd = TPE(dataset,"target",task_type,algo_type)
                sgd.fit(x_train, y_train)
                predicted = sgd.predict(x_test)#X or x_test
                result = pd.DataFrame(predicted)
                result.to_csv('./result/SGD_Regressor_result.csv', index=False, header=True)
                
            elif choose == "Y":
                print("")
                print("few features should be important")
                choose = input("Enter Y or N ")
                if choose == "N":
                    print("----------------------------------------------------------")
                    #print("RidgeRegression")
                    print("Linear SVR")
                    print("----------------------------------------------------------")
                    algo_type = LinearSVR() 
                    task_type = 'reg'                  
                    lin_svr = TPE(dataset,"target",task_type,algo_type)
                    lin_svr.fit(x_train, y_train)
                    predicted = lin_svr.predict(x_test)#X or x_test
                    result = pd.DataFrame(predicted)
                    result.to_csv('./result/LinearSVR_result.csv', index=False, header=True)
                    print("")
                    choose = input("If it doesn't work, press Y. ")
                    if choose == "Y":
                        print("----------------------------------------------------------")
                        print("SVR(kernel=\'rbf\')") #SVR default
                        print("EnsembleRegressors")
                        print("----------------------------------------------------------")
                elif choose =="Y":
                    print("----------------------------------------------------------")
                    print("Lasso")
                    print("ElasticNet")
                    print("----------------------------------------------------------")
    elif choose == "Y":
        print("")
        print("do you have labeled data")
        choose = input("Enter Y or N ")
        if choose == "N":
            print("")
            print("number of categories Known")
            choose = input("Enter Y or N ")
            if choose == "N":
                print("")
                print("<10K samples")
                if data_size < 10000 : 
                    Yes()
                else :
                    No()
                if choose == "N":
                    print("----------------------------------------------------------")
                    print("tough luck")
                    print("----------------------------------------------------------")
                elif choose == "Y":
                    print("----------------------------------------------------------")
                    print("MeanShift")
                    print("VBGMM")
                    print("----------------------------------------------------------")
            elif choose == "Y":
                print("")
                print("<10K samples")
                if data_size < 10000 : 
                    Yes()
                else :
                    No()
                if choose == "N":
                    print("----------------------------------------------------------")
                    print("MiniBath Kmeans")
                    print("----------------------------------------------------------")
                elif choose == "Y":
                    print("----------------------------------------------------------")
                    print("KMeans")
                    print("----------------------------------------------------------")
                    '''
                    f = open("./result/Kmeans.py", "w")
                    code = "#Import Library\nfrom sklearn.cluster import KMeans\n#Assumed you have, X for training data set\n#and x_test of test_dataset\n#Create KNeighbors classifier object model\nk_means = KMeans(n_clusters=3, random_state=0)\n#Train the model using the training sets and check score\nk_means.fit(X)\n#predict output\npredicted = k_means.predict(X)#X or x_test"
                    f.write(code)
                    f.close()
                    f = open("./result/Kmeans.txt", "w")
                    code = "#Import Library\nfrom sklearn.cluster import KMeans\n#Assumed you have, X for training data set\n#and x_test of test_dataset\n#Create KNeighbors classifier object model\nk_means = KMeans(n_clusters=3, random_state=0)\n#Train the model using the training sets and check score\nk_means.fit(X)\n#predict output\npredicted = k_means.predict(X)#X or x_test"
                    f.write(code)
                    f.close()
                    print("")
                    '''
                    from sklearn.cluster import KMeans
                    X = dataset.copy()
                    k_means = KMeans(n_clusters=3, random_state=0)
                    #Train the model using the training sets and check score
                    k_means.fit(X)
                    #predict output
                    predicted = k_means.predict(X)#X or x_test
                    result = pd.DataFrame(predicted)
                    result.to_csv('./result/Kmeans_result.csv', index=False, header=True)
                    
                    choose = input("If it doesn't work, press Y. ")
                    if choose == "Y":
                        print("----------------------------------------------------------")
                        print("Spectral Clustreing")
                        print("GMM")
                        print("----------------------------------------------------------")
        elif choose == "Y":
            print("")
            print("<100K samples")
            if data_size < 100000 : 
              Yes()
            else :
              No()
            if choose == "N":
                print("----------------------------------------------------------")
                print("SGD Classifier")
                print("----------------------------------------------------------")
                print("")
                choose = input("If it doesn't work, press Y. ")
                if choose == "Y":
                    print("----------------------------------------------------------")
                    print("kernel approximation")
                    print("----------------------------------------------------------")
            elif choose == "Y":
                print("----------------------------------------------------------")
                print("Linear SVC")
                print("----------------------------------------------------------")
                print("")
                choose = input("If it doesn't work, press Y. ")
                if choose == "Y":
                    print("")
                    print("Text Data")
                    if str_type:
                      Yes()
                    else :
                      No()
                    if choose == "N":
                        print("----------------------------------------------------------")
                        print("KNeighbors Classifier")
                        print("----------------------------------------------------------")
                        print("")
                        choose = input("If it doesn't work, press Y. ")
                        if choose == "Y":
                            print("----------------------------------------------------------")
                            print("SVC")
                            print("Ensemble Classifiers")
                            print("----------------------------------------------------------")
                    elif choose == "Y":
                        print("----------------------------------------------------------")
                        print("Naive Bayes")
                        print("----------------------------------------------------------")


time.sleep(9999)
