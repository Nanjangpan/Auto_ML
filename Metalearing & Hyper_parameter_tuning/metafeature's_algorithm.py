import numpy as np
import pandas as pd
import math
import scipy.stats
from scipy.stats import kurtosis
import pickle

import warnings
 
warnings.filterwarnings("ignore")

df = pd.read_csv("../Dataset/titanic_Survived.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
meta_data = []

col = [] 

def NumberOfInstances(X,y):
    return X.shape[0]
meta_data.append(NumberOfInstances(X,y))

def NumberOfFeatures(X,y):
   return X.shape[1]
meta_data.append(NumberOfFeatures(X,y))

def NumberOfMissingValues(X,y):
   count = 0
   for i in range(0, X.shape[0]):
      if X.iloc[i,:].isnull().sum() == 0:
         count += 0
      else:
         count += X.iloc[i,:].isnull().sum()
   return count
meta_data.append(NumberOfMissingValues(X,y))

def PercentageOfMissingValues(X, y):
   total = X.shape[0] * X.shape[1]
   return NumberOfMissingValues(X,y)/total
meta_data.append(PercentageOfMissingValues(X, y))

def NumberOfInstancesWithMissingValues(X,y):
   count = 0
   for i in range(0, X.shape[0]):
      if X.iloc[i,:].isnull().sum() == 0:
         count += 0
      else:
         count += 1
   return count
#meta_data.append(NumberOfInstancesWithMissingValues(X,y))

def PercentageOfInstancesWithMissingValues(X,y):
   return NumberOfInstancesWithMissingValues(X,y)/NumberOfInstances(X,y)
#meta_data.append(PercentageOfInstancesWithMissingValues(X,y))

def NumberOfFeaturesWithMissingValues(X,y):
   count = 0
   for i in range(0, X.shape[1]):
      if X.iloc[:,i].isnull().sum() == 0:
         count += 0
      else:
         count += 1
   return count
meta_data.append(NumberOfFeaturesWithMissingValues(X,y))

def PercentageOfFeaturesWithMissingValues(X,y):
   return NumberOfFeaturesWithMissingValues(X,y)/NumberOfFeatures(X,y)
meta_data.append(PercentageOfFeaturesWithMissingValues(X,y))

def NumberOfNumericFeatures(X,y):
    try:
        temp = [x for x in X.get_dtype_counts().keys() if 'int' in x or 'float' in x]
        cnt = 0
        for i in temp:
            cnt += df.get_dtype_counts()[i]
        return cnt
    except:
        print("unexpected error")
meta_data.append(NumberOfNumericFeatures(X,y))

def NumberOfCategoricalFeatures(X,y):
    try:
        return X.dtypes.value_counts()['object']
    except KeyError:
        return 0
    except:
        print("unexpected error")
#meta_data.append(NumberOfCategoricalFeatures(X,y))
      
# /0의 nan return
def RatioNumericalToNominal(X,y):
    #according to: https://arxiv.org/pdf/1808.03233.pdf
    #the ratio of number of numerical features to the number of categorical features
    temp0 =  NumberOfNumericFeatures(X,y)
    temp1 =  NumberOfCategoricalFeatures(X,y)
    if temp1 == 0:
        return 0
    return temp0/temp1
#meta_data.append(RatioNumericalToNominal(X,y))

# /0의 nan return
def RatioNominalToNumerical(X,y):
    temp0 =  NumberOfNumericFeatures(X,y)
    temp1 =  NumberOfCategoricalFeatures(X,y)
    if temp0 == 0:
        return np.nan 
    return temp1/temp0
#meta_data.append(RatioNominalToNumerical(X,y))

def DatasetRatio(X,y):
    #according to: https://arxiv.org/pdf/1808.03233.pdf
    #the ratio of number of features to the number of data points
    return X.shape[1]/X.shape[0]
#meta_data.append(DatasetRatio(X,y))

def LogDatasetRatio(X,y):
    return math.log(DatasetRatio(X,y))
#meta_data.append(LogDatasetRatio(X,y))

def InverseDatasetRatio(X,y):
    return 1/DatasetRatio(X,y)
#meta_data.append(InverseDatasetRatio(X,y))

def LogInverseDatasetRatio(X,y):
    return math.log(InverseDatasetRatio(X,y))
#meta_data.append(LogInverseDatasetRatio(X,y))

def NumberOfBinaryFeatures(X,y):
    try:
        temp = []
        for col in X.columns:
            if len(X[col].value_counts()) == 2:
                temp.append(col)
        return X[temp].dtypes.value_counts()['object']
    except KeyError:
        return 0
    except:
        print("unexpected error")
meta_data.append(NumberOfBinaryFeatures(X,y))
    
def NumberOfClasses(X, y):
    temp = set(y)
    return len(temp)
meta_data.append(NumberOfClasses(X, y))

def NumberOfSymbolicFeatures(X,y):
    try:
        return X.dtypes.value_counts()['object']*NumberOfInstances(X,y)
    except KeyError:
        return 0
    except:
        print("unexpected error")
meta_data.append(NumberOfSymbolicFeatures(X,y))

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
def Skewnesses(X, y):
    skews = []
    for i in range(X.shape[1]):
        skews.append(scipy.stats.skew(X[:, i].astype(np.float)))

    return skews

def SkewnessMin(X, y):
    skews = Skewnesses(X, y)
    minimum = np.nanmin(skews) if len(skews) > 0  else 0
    return minimum if np.isfinite(minimum) else 0
#meta_data.append(SkewnessMin(X, y))

def SkewnessMax(X, y):
    skews = Skewnesses(X, y)
    maximum = np.nanmax(skews) if len(skews) > 0  else 0
    return maximum if np.isfinite(maximum) else 0
#meta_data.append(SkewnessMax(X, y))

def SkewnessMean(X, y):
    skews = Skewnesses(X, y)
    mean = np.nanmean(skews) if len(skews) > 0  else 0
    return mean if np.isfinite(mean) else 0
#meta_data.append(SkewnessMean(X, y))

def Kurtosisses(X, y):
    kurts = []
    for i in range (X.shape[1]):
        kurts.append(scipy.stats.kurtosis (X[:, i].astype(np.float)))
    return kurts

def KurtosisMin(X, y):
    kurts = Kurtosisses(X, y)
    minimum = np.nanmin(kurts) if len(kurts) > 0  else 0
    return minimum if np.isfinite(minimum) else 0
#meta_data.append(KurtosisMin(X, y))

def KurtosisMax(X, y):
    kurts = Kurtosisses(X, y)
    maximum = np.nanmax(kurts) if len(kurts) > 0  else 0
    return maximum if np.isfinite(maximum) else 0
#meta_data.append(KurtosisMax(X, y))

def KurtosisMean(X, y):
    kurts = Kurtosisses(X, y)
    mean = np.nanmean(kurts) if len(kurts) > 0  else 0
    return mean if np.isfinite(mean) else 0
#meta_data.append(KurtosisMean(X, y))

#print(meta_data)

#load 25_property_dataframe -> sample data
with open('25_property_dataframe.pickle', 'rb') as f:
  df = pickle.load(f)
  df.fillna(0, inplace=True)

null_list = ['NumberOfFeaturesWithMissingValues', 'PercentageOfFeaturesWithMissingValues', 'NumberOfCategoricalFeatures', 'RatioNumericalToNominal', 'RatioNominalToNumerical', 'DatasetRatio', 'LogDatasetRatio', 'InverseDatasetRatio', 'LogInverseDatasetRatio', 'SkewnessMin',
             'SkewnessMax', 'SkewnessMean',  'KurtosisMin', 'KurtosisMax', 'KurtosisMean']

df.drop(null_list, inplace=True, axis=1)

#find max cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity

meta_array = np.array([meta_data])
'''
cos_sim_arr = []
cos = 0
for i in range(0, df.shape[0]): 
    vec = list(df.iloc[i,:])
    for j in range(0, len(vec)):
       cos+=

'''
cos_sim_arr = []
for i in range(0, df.shape[0]):
    vec = np.array([list(df.iloc[i,:])])
    tmp = cosine_similarity(meta_array, vec)[0][0]
    cos_sim_arr.append([tmp, i])

max_cos_val = max(cos_sim_arr)
max_cos_idx = cos_sim_arr[cos_sim_arr.index(max(cos_sim_arr))][1]
print(max_cos_val[0])
print(max_cos_idx)




