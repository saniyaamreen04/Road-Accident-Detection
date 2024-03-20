# Importing the libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.model_selection import GridSearchCV 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score, roc_auc_score, roc_LANE 
# Importing the dataset 
dataset = pd.read_csv('../Dataset/diabetes.csv') 
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, 8].values 
# Splitting the dataset into the Training set and Test set 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42) 
# Feature Scaling 
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() 
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test) 
# Parameter evaluation 
treeclf = DecisionTreeClassifier(random_state=42) 
parameters = {'max_depth': [6, 7, 8, 9], 'min_samples_split': [2, 3, 4, 5], 'max_features': 
[1,2, 3, 4] 
} 
gridsearch=GridSearchCV(treeclf, parameters, cv=100, scoring='roc_auc') 
gridsearch.fit(X,y) 
print(gridsearch.best_params_) 
print(gridsearch.best_score_) 
# Adjusting development threshold 
tree = DecisionTreeClassifier(max_depth = 6, max_features = 4, min_samples_split = 5, 
random_state=42) 
X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=42) 
tree.fit(X_train, y_train) 
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train))) 
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test))) 
# Predicting the Test set results 
y_pred = tree.predict(X_test)  
# Making the Confusion Matrix 
from sklearn.metrics import classification_report, confusion_matrix 
cm = confusion_matrix(y_test, y_pred) 
print('TP - True Negative {}'.format(cm[0,0])) 
print('FP - False Positive {}'.format(cm[0,1])) 
print('FN - False Negative {}'.format(cm[1,0])) 
print('TP - True Positive {}'.format(cm[1,1])) 
print('Accuracy Rate: {}'.format(np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm)))) 
print('MisclassificationRate:{}'.format(np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm)) 
)) 
round(roc_auc_score(y_test,y_pred),5)
# Importing the libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score, roc_auc_score, roc_LANE 
from sklearn.neighbors import KNeighborsClassifier 
# Importing the dataset 
dataset = pd.read_csv('../Dataset/diabetes.csv') 
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, 8].values 
# Splitting the dataset into the Training set and Test set 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,                                                  
random_state = 42) 
# Feature Scaling 
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() 
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test) 
# Parameter evaluation 
knnclf = KNeighborsClassifier() 
parameters={'n_neighbors': range(1, 20)} 
gridsearch=GridSearchCV(knnclf, parameters, cv=100, scoring='roc_auc') 
gridsearch.fit(X, y) 
print(gridsearch.best_params_) 
print(gridsearch.best_score_)
# Fitting K-NN to the Training set 
knnClassifier = KNeighborsClassifier(n_neighbors = 18) 
knnClassifier.fit(X_train, y_train) 
print('Accuracy of K-NN classifier on training 
set:{:.2f}'.format(knnClassifier.score(X_train, y_train))) 
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knnClassifier.score(X_test, 
y_test))) 
# Predicting the Test set results 
y_pred = knnClassifier.predict(X_test) 
# Making the Confusion Matrix 
from sklearn.metrics import classification_report, confusion_matrix 
cm = confusion_matrix(y_test, y_pred) 
print('TP - True Negative {}'.format(cm[0,0])) 
print('FP - False Positive {}'.format(cm[0,1])) 
print('FN - False Negative {}'.format(cm[1,0])) 
print('TP - True Positive {}'.format(cm[1,1])) 
print('Accuracy Rate: {}'.format(np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm)))) 
print('MisclassificationRate:{}'.format(np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))
 )) 
round(roc_auc_score(y_test,y_pred),5) 
# Importing the libraries 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score, roc_auc_score, roc_LANE 
from sklearn.svm import SVC 
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report
# Importing the dataset 
dataset = pd.read_csv('../Dataset/diabetes.csv') 
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, 8].values 
# Splitting the dataset into the Training set and Test set 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,                                                     
random_state = 42) 
# Feature Scaling 
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() 
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test) 
#svm with grid search 
svm = SVC(random_state = 42) 
parameters = {'kernel':('linear', 'rbf'), 'C':(1,0.25,0.5,0.75), 
'gamma': (1,2,3,'auto'),'decision_function_shape':('ovo','ovr'), 
'shrinking':(True,False)} 
scores = ['precision', 'recall'] 
for score in scores: 
print("# Tuning hyper-parameters for %s" % score) 
print() 
svm = GridSearchCV(SVC(), parameters, cv=5, 
scoring='%s_macro' % score) 
svm.fit(X_train, y_train) 
print("Best parameters set found on development set:") 
print() 
print(svm.best_params_) 
print()
print("Grid scores on development set:") 
print() 
means = svm.cv_results_['mean_test_score'] 
stds = svm.cv_results_['std_test_score'] 
for mean, std, params in zip(means, stds, svm.cv_results_['params']): 
print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params)) 
print() 
print("Detailed classification report:") 
print() 
print("The model is trained on the full development set.") 
print("The scores are computed on the full evaluation set.") 
print() 
y_true, y_pred = y_test, svm.predict(X_test) 
print(classification_report(y_true, y_pred)) 
print() 
svm_model = SVC(kernel='rbf', C=100, gamma = 0.0001, random_state=42) 
svm_model.fit(X_train, y_train) 
spred = svm_model.predict(X_test) 
print ('Accuracy with SVM {0}'.format(accuracy_score(spred, y_test) * 100)) 
# Making the Confusion Matrix 
from sklearn.metrics import classification_report, confusion_matrix 
cm = confusion_matrix(y_test, y_pred) 
print('TP - True Negative {}'.format(cm[0,0])) 
print('FP - False Positive {}'.format(cm[0,1])) 
print('FN - False Negative {}'.format(cm[1,0])) 
print('TP - True Positive {}'.format(cm[1,1])) 
print('Accuracy Rate: {}'.format(np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm)))) 
print('MisclassificationRate:{}'.format(np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))
 )) 
svm.fit(X_train, y_train) 
round(roc_auc_score(y_test,y_pred),5)
