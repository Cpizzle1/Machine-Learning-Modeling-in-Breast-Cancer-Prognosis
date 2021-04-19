from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pandas as pd 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error, accuracy_score, f1_score,accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.svm import SVC

from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import recall_score, precision_score, roc_curve, auc, make_scorer, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from collections import defaultdict
# from roc import plot_roc
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

    # train_filepath = 'data/X_train.csv'
    # test_filepath = 'data/X_test.csv'
    

    # gradient_boosting_grid = {'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5]
    #                          ,'max_depth': [2, 4, 8]
    #                          ,'subsample': [0.25, 0.5, 0.75, 1.0]
    #                          ,'min_samples_leaf': [1, 2, 4]
    #                          ,'max_features': ['sqrt', 'log2', None]
    #                          ,'n_estimators': [5,10,25,50,100,200]}

    # random_forest_grid = {'max_depth': [2, 4, 8]
    #                      ,'max_features': ['sqrt', 'log2', None]
    #                      ,'min_samples_leaf': [1, 2, 4]
    #                      ,'min_samples_split': [2, 4]
    #                      ,'bootstrap': [True, False]
    #                      ,'n_estimators': [5,10,25,50,100,200]}

    # gradient_randomsearch = RandomizedSearchCV(GradientBoostingClassifier()
    #                                           ,gradient_boosting_grid
    #                                           ,n_jobs=-1
    #                                           ,verbose=False
    #                                           ,scoring='roc_auc')

    # random_foreset_randomsearch = RandomizedSearchCV(RandomForestClassifier()
    #                                                 ,random_forest_grid
    #                                                 ,n_jobs=-1
    #                                                 ,verbose=False
    #                                                 ,scoring='roc_auc')
    # svm_randomsearch = RandomizedSearchCV(SVC(gamma = 'auto')
    #                                           ,svm_grid
    #                                           ,n_jobs=-1
    #                                           ,verbose=False
    #                                           ,scoring='roc_auc')
    # logistic_regression_grid = {'Cs':[2, 5]
    #                        ,'cv':[2,4,8]
    #                        ,'solver':['lbfgs', 'liblinear']
    #                        ,'max_iter' : [50]
                            
    #                        }

    # gradient_randomsearch.fit(X_train, y_train)
    # print(f"Best Gradient Parameters: {gradient_randomsearch.best_params_}")
    # print(f"Best Gradient Model: {gradient_randomsearch.best_estimator_}")
    # print(f"Best Gradient Score: {gradient_randomsearch.best_score_:.4f}")

    # random_foreset_randomsearch.fit(X_train, y_train)
    # print(f"Best Random Forest Parameters: {random_foreset_randomsearch.best_params_}")
    # print(f"Best Random Forest Model: {random_foreset_randomsearch.best_estimator_}")
    # print(f"Best Random Forest Score: {random_foreset_randomsearch.best_score_:.4f}")

    # X_test, y_test, test_df = final_clean(test_filepath, cols_to_int)
    
    # gradient_best_model = gradient_randomsearch.best_estimator_
    # random_forest_best_model = random_foreset_randomsearch.best_estimator_

    # gy_hats = gradient_best_model.predict(X_test)
    # ry_hats = random_forest_best_model.predict(X_test)

    # print(f"Gradient ROC Score = {roc_auc_score(y_test, gy_hats):.4f}")
    # print(f"Random Forest ROC Score = {roc_auc_score(y_test, ry_hats):.4f}")




def optimize_model(classifier, param_grid, X_train, X_test, y_train, y_test, scoring):
    GS = GridSearchCV(classifier, param_grid, scoring=scoring, verbose=10, cv=2, refit=False)
    GS.fit(X_train2, y_train2)
    return GS.cv_results_ 
def specificity(y_test, y_pred):
    TN = np.sum([(y_pred==0) & (y_test==0)])
    FP = np.sum([(y_pred==1) & (y_test==0)])
    return TN/(TN+FP)

def neg_pred(y_test, y_pred):
    TN = np.sum([(y_pred==0) & (y_test==0)])
    FN = np.sum([(y_pred==0) & (y_test==1)])
    return TN/(TN+FN)

if __name__ == '__main__':
    # scoring = 'accuracy'

    # y = pd.read_csv('~/data/target_death_col.csv')

    # y = y.reshape(-1)
    
    # X = pd.read_csv('~/data/capstone2.mrn_df2_laptop.csv')
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=1, stratify = y)
    # X_train2, X_test, y_train2, y_test = train_test_split(X_train.copy(), y_train.copy(), test_size=0.10, random_state=1)

    # tree = DecisionTreeClassifier(class_weight='balanced')
    
    # ada = AdaBoostClassifier()
    # param_grid_ada = {'base_estimator': [tree], \
    #                   'n_estimators': [100, 250, 500], \
    #                   'learning_rate': [0.1, 0.25, 0.5, 0.75, 1.0]}

    # rfc = RandomForestClassifier()
    # param_grid_rfc = {'n_estimators': [1000, 5000, 10000, 15000], \
    #                   'n_jobs': [-1], \
    #                   'max_features': [10, 50, 100, 200], \
    #                   'max_depth': [3, 4, 5, 6], \
    #                   'class_weight': ['balanced']}

    # svc = SVC()
    # param_grid_svc = {'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5], \
    #                   'gamma': [50, 100, 150, 200], \
    #                   'class_weight': ['balanced']}



    # SGD = SGDClassifier()
    # param_grid_sgd = {'loss': ['hinge', 'log', 'modified_huber'], \
    #                   'alpha': [0.001, 0.01, 0.1, 1.0], \
    #                   'penalty': ['l1'], \
    #                   'max_iter': [5, 10, 25, 50, 75], \
    #                   'class_weight': ['balanced']}

    # param_list = [param_grid_ada, param_grid_rfc, param_grid_sgd, param_grid_svc]
    # clf_list = [ada, rfc, SGD, svc]

    # results_list= []
    # for i, classifier in enumerate(clf_list):
    #     results = optimize_model(clf_list[i], param_list[i], X_train, X_test, y_train, y_test, scoring)
    #     results_list.append(results)
    # print(results_list)
