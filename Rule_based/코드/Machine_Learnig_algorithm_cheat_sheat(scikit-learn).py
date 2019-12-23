# scikit-learn algorithm cheat-sheet
import time
import csv
import numpy as np
import pandas as pd

dataset = pd.read_csv('./data/iris.csv', engine='python')
data_size = dataset.shape[0]
a = dataset.iloc[:, -1][1]
if(issubclass(type(a), str)): #check predict data is stirng
  str_type = True
else :
  str_type = False

choose = ''

def No():
  global choose
  choose = "N"
  print("No\n")
  time.sleep(1)
    
def Yes():
  global choose
  choose = "Y"
  print("Yes\n")
  time.sleep(1)

def Oversampling():
  from sklearn.datasets import make_classification
  from sklearn.decomposition import PCA
  from imblearn.over_sampling import SMOTE
  # 모델설정
  sm = SMOTE(ratio='auto', kind='regular')
  x_train, y_test = sm.fit_sample(x_train,list(y_train))

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
            elif choose == "Y":
                print("")
                print("few features should be important")
                choose = input("Enter Y or N ")
                if choose == "N":
                    print("----------------------------------------------------------")
                    print("RidgeRegression")
                    print("SVR(kernel=\'linear\')")
                    print("----------------------------------------------------------")
                    print("")
                    choose = input("If it doesn't work, press Y. ")
                    if choose == "Y":
                        print("----------------------------------------------------------")
                        print("SVR(kernel=\'rbf\')")
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
