# scikit-learn algorithm cheat-sheet
import time
import csv
import numpy as np
import pandas as pd
import warnings
 
warnings.filterwarnings("ignore")



#import candidate models
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
#from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from missingpy import MissForest
from math import sqrt




#preprocess
#-----------------------------------------------------------------------------------------------------------------------
def preprocess4ensemble(df):
    print("preprocess4ensemble")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_num = X.select_dtypes(include="number")
    X_cat = X.select_dtypes(include="object")

    #Check if all right
    if not(X_num.shape[1]+X_cat.shape[1] == X.shape[1]) or not(X_num.shape[0] == X_cat.shape[0] and  X_cat.shape[0] == X.shape[0]):
        print("categorical and numerical seperation operation has a problem")

    for cat_col in X_cat.columns:
        X_cat = pd.concat([X_cat,pd.get_dummies(X_cat[cat_col],dummy_na=False)],axis=1)
        del X_cat[cat_col]

    imputer = MissForest()
    X_imputed = pd.DataFrame(imputer.fit_transform(X_num))
    X_imputed.columns = X_num.columns
    del X_num

    X = pd.concat([X_imputed,X_cat],axis=1)
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)
    return X_res, y_res

def preprocess4xgb(df):
    print("preprocess4xgb")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_num = X.select_dtypes(include="number")
    X_cat = X.select_dtypes(include="object")

    #Check if all right
    if not(X_num.shape[1]+X_cat.shape[1] == X.shape[1]) or not(X_num.shape[0] == X_cat.shape[0] and  X_cat.shape[0] == X.shape[0]):
        print("categorical and numerical seperation operation has a problem")

    for cat_col in X_cat.columns:
        X_cat = pd.concat([X_cat,pd.get_dummies(X_cat[cat_col],dummy_na=False)],axis=1)
        del X_cat[cat_col]

#     imputer = MissForest()
#     X_imputed = pd.DataFrame(imputer.fit_transform(X_num))
#     X_imputed.columns = X_num.columns
#     del X_num
    def oversampling(X_train,y_train):
        rus = RandomOverSampler(return_indices=True)
        X_resampled, y_resampled, idx_resampled = rus.fit_resample(X_train, y_train)
        X_resampled = pd.DataFrame(X_resampled)
        y_resampled = pd.Series(y_resampled)
        X_resampled.columns = X_train.columns
        return X_resampled,y_resampled
    
    X = pd.concat([X_imputed,X_cat],axis=1)
    print(type(X))
    ros = RandomOverSampler(random_state=42)
    X, y = X.fillna(10000000000), y
    X, y = oversampling(X, y)
    X, y = X.replace(10000000000, np.nan), y.replace(10000000000, np.nan)
    
    return X, y

def preprocess4normal(df):
    print("preprocess4normal")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_num = X.select_dtypes(include="number")
    X_cat = X.select_dtypes(include="object")

    #Check if all right
    if not(X_num.shape[1]+X_cat.shape[1] == X.shape[1]) or not(X_num.shape[0] == X_cat.shape[0] and  X_cat.shape[0] == X.shape[0]):
        print("categorical and numerical seperation operation has a problem")

    for cat_col in X_cat.columns:
        X_cat = pd.concat([X_cat,pd.get_dummies(X_cat[cat_col],dummy_na=False)],axis=1)
        del X_cat[cat_col]

    #Impute nan
    imputer = MissForest()
    X_imputed = pd.DataFrame(imputer.fit_transform(X_num))
    X_imputed.columns = X_num.columns
    del X_num
    
    #Scaling
    scaled_features = StandardScaler().fit_transform(X_imputed.values)
    scaled_features_df = pd.DataFrame(scaled_features, index=X_imputed.index, columns=X_imputed.columns)

    X = pd.concat([scaled_features_df,X_cat],axis=1)
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)
    return X_res, y_res

def preprocess(df,algo_type):
    if "RandomForest" in str(algo_type).split("(")[0] or "GradientBoosting" in str(algo_type).split("(")[0]:
        return preprocess4ensemble(df)
    elif "XGB" in str(algo_type).split("(")[0]:
        return preprocess4xgb(df)
    else:
        return preprocess4normal(df)
      
#-----------------------------------------------------------------------------------------------------------------------
      
dataset = pd.read_csv("../Dataset/housePrice_SalePrice.csv")
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
  print("No")
  time.sleep(1)

#if Yes, we use this function    
def Yes():
  global choose
  choose = "Y"
  print("Yes")
  time.sleep(1)


#-----------------------------------------------------------------------------------------------------------------------
  
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
                sgd = SGDRegressor()
                result = preprocess(dataset, sgd)
                x = pd.DataFrame(result[0])
                y = pd.DataFrame(result[1])
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
                sgd.fit(x_train, y_train)
                predicted = sgd.predict(x_test)
                mse=mean_squared_error(y_test, predicted)
                rmse=sqrt(mse)
                print("rmse: %.2f" %rmse)
                #print("Accuracy: %.2f" %accuracy_score(y_test, predicted))
                #print("F1 score: %.2f" %f1_score(y_test, predicted))
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
                    lin_svr = LinearSVR (
                                epsilon=0,
                                tol=0.0001,
                                C=1,
                                fit_intercept=True,
                                intercept_scaling=1,
                                dual=True,
                                verbose=0,
                                random_state=None,
                                max_iter=1000
                    )
                    result = preprocess(dataset, lin_svr)
                    x = pd.DataFrame(result[0])
                    y = pd.DataFrame(result[1])
                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
                    lin_svr.fit(x_train, y_train)
                    predicted = lin_svr.predict(x_test)
                    mse=mean_squared_error(y_test, predicted)
                    rmse=sqrt(mse)
                    print("rmse: %.2f" %rmse)
                    result = pd.DataFrame(predicted)
                    result.to_csv('./result/LinearSVR_result.csv', index=False, header=True)
                    print("")
                    choose = input("If it doesn't work, press Y. ")
                    if choose == "Y":
                        print("----------------------------------------------------------")
                        #print("SVR(kernel=\'rbf\')") #SVR default
                        print("EnsembleRegressors")
                        print("----------------------------------------------------------")
                        clf = RandomForestRegressor(
                                  bootstrap = False,
                                  max_depth = None,
                                  max_features= 'sqrt',
                                  max_leaf_nodes = None,
                                  min_impurity_decrease = 0.0,
                                  min_impurity_split = None,
                                  min_samples_leaf = 1,
                                  min_samples_split = 2,
                                  min_weight_fraction_leaf = 0.0,
                                  n_estimators = 1500,
                                  n_jobs = 1,
                                  oob_score = False,
                                  random_state = 42,
                                  verbose = 0,
                                  warm_start = False) 
                        result = preprocess(dataset, clf)
                        x = pd.DataFrame(result[0])
                        y = pd.DataFrame(result[1])
                        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
                        clf.fit(x_train, y_train)
                        predicted = clf.predict(x_test)
                        mse=mean_squared_error(y_test, predicted)
                        rmse=sqrt(mse)
                        print("rmse: %.2f" %rmse)
                        result = pd.DataFrame(predicted)
                        result.to_csv('./result/clf_result.csv', index=False, header=True)
                        print("")
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
