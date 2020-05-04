# Auto_ML
Rule based Auto ML 과 Meta Learning based Auto ML 두가지를 구현해보는 프로젝트
<br>
## Rule based Auto ML
Scikit learn에서 제공해준 [Machine learning cheat sheet](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)기반으로 구현하였습니다. 


* Rule based Auto ML Pipeline
<img width="559" alt="Rule-based" src="https://user-images.githubusercontent.com/50066454/80966861-4f77ca80-8e50-11ea-9231-4eaf93026061.PNG">


## Meta Learning based Auto ML
Meta Learning 기반으로 Auto ML을 구현하였습니다. <br>


* Meta Learning based Auto ML Pipeline <br>
<img width="551" alt="data-preprocessing" src="https://user-images.githubusercontent.com/50066454/80968559-23117d80-8e53-11ea-9e36-eca94785cf19.png"></img>

* Meta Learning
  * 목표 : 최적의 Model 선정
  * Open ML의 수많은 dataset들의 meta feature 보유
  * 새로운 데이터의 meta feature 계산하여 Open ML내 datasets들의 meta feature 비교(Cosine similarity 이용)
  * 데이터에 맞는 최적의 model 5개 선정
* Data pre-processing <br>
&emsp;<img width="551" alt="data-preprocessing" src="https://user-images.githubusercontent.com/50066454/80968559-23117d80-8e53-11ea-9e36-eca94785cf19.png"></img>
* Bayesian optimizer
  * 목표 : Model 별 최적의 hyper-parameter 선정
  * Surrogate Model : TPE 사용
  * Meta Learning 단계에서 선별 된 5가지 Model 별 최적의 hyper-parameter 선정
 
## 프로젝트 결과
Heart Disease Prediciton Classification([Dataset](https://www.kaggle.com/ronitf/heart-disease-uci))
* Rule based
  * Accuracy : 0.85
  * F1 score : 0.84

* Meta Learning based
  * Accuracy : 0.89
  * F1 score : 0.88
  
