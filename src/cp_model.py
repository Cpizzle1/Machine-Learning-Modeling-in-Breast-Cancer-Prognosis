from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import  roc_auc_score, roc_curve, auc,  accuracy_score,f1_score, accuracy_score, precision_score, recall_score, classification_report
import numpy as np
import pandas as pd 
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV, SGDClassifier
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier,GradientBoostingClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
import pickle
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def optimize_model2_randomCV(model, grid_params, X_train, y_train, scoring):
    """[Takes a model in, grid parameters, X and y values returns  best model found using the input parameters]

    Args:
        model ([Classifcation model sklearn]): [Logistic Regression, Random Forrest, Gradient Boost, and others]
        grid_params ([dictionary]): [keys are strings of parater, values are list of values to try]
        X_train ([Pandas dataframe]): [training feature data]
        y_train ([numpy array]): [array of target values]
        scoring ([scoring type to measure]): [sklearn scoring options for given model]

    Returns:
        [type]: [description]
    """

    model_search = RandomizedSearchCV(model
                                        ,grid_params
                                        ,n_jobs=-1
                                        ,verbose=False
                                        ,scoring=scoring)
    model_search.fit(X_train, y_train)
    print(f"Best Parameters for {model}: {model_search.best_params_}")
    print(f"Best Model for {model}: {model_search.best_estimator_}")
    print(f"Best Score for {model}: {model_search.best_score_:.4f}")
    
    return model_search.best_estimator_

def best_model_predictor(model, X_test, y_test):
    """[returns Analysis of model on test set or valudation set]

    Args:
        model ([sklearn classifer model]): [Logistic regression, Random Forrest, Gradient Boosting, etc]
        X_test ([Pandas dataframe]): [Test feature data]
        y_test ([numpy array]): [target valudation data]
    """

    
    y_hats = model.predict(X_test)
    print(f"{model} ROC Score = {roc_auc_score(y_test, y_hats):.3f}")
    print(f"{model} F1 Score = {f1_score(y_test, y_hats):.3f}")
    print(f"{model} Accuracy Score = {accuracy_score(y_test, y_hats):.3f}")
    print(classification_report(y_test, y_hats))


def roc_curve_grapher(model, X_test ,y_test):
    """[Makes ROC curve graph given model and data]

    Args:
        model ([SKlearn classifer model]): [Logistic regression, Random Forrest, Gradient Boosting, etc]]
        X_test ([Pandas dataframe]): [Test feature data]
        y_test ([numpy array]): [target valudation data]
    """

    yhat = model.predict_proba(X_test)
    yhat = yhat[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, yhat)
    plt.plot([0,1], [0,1], linestyle='--', label='Random guess')
    plt.plot(fpr, tpr, marker='.', label=f'Model')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.suptitle('Model ROC curve', fontsize=20)
    plt.legend()
    plt.savefig("Logistic Regression_ROC_curve.png", dpi=200)
    plt.show()

def beta_grapher_for_log_regressors_VIF_version(df, model):
    """[graphs Model coefficents of Lasso regression for intpretation]

    Args:
        df ([Training features dataframe]): [X_train information use to create model]
        model ([Lasso Regression SKLearn model]): [logisticRegressor w/ penalty = l1]
    """
    lst_coefs = []
    for name, coef in zip(df.columns[1:], model.coef_[0]):
        lst_coefs.append((name, coef))
       

    vif_dict = {}
    for i in lst_coefs:
        if i[1] != 0:
            vif_dict[i[0]] = i[1]
    vif_df_ordered = pd.DataFrame.from_dict(vif_dict, orient='index')
    vif_df_ordered.sort_values(by =0, inplace = True)
    vif_df_ordered[1] = vif_df_ordered.index
    vif_df_ordered.plot.bar(figsize = (16,10))
    vif_df_ordered.plot(legend=None)
    plt.xlabel("Genes", fontsize = 13)
    plt.ylabel("Coefficient values", fontsize = 13)
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
   
    
    plt.show()


def threshold_recall_scorer(model, threshold_list):
    results = {}
    for threshold in threshold_list:
        yhat_probs = model.predict_proba(X_test)[:,1]
        preds = np.where(yhat_probs > threshold, 1, 0)
        results[f'threshold {threshold}'] = [('F1 Score',round(f1_score(y_test,preds), 2)), 
                                        ('Accuracy Score', round(accuracy_score(y_test,preds),2)),
                                        ('AUC Score', round(roc_auc_score(y_test,preds),2)),\
                                        ('Recall Score', round(recall_score(y_test,preds),2))]
    print(results)


if __name__ == '__main__':


    X_train = pd.read_csv('/Users/cp/Documents/dsi/capstone2/capstone2/data/train_setVIF_wednesday_features.csv')
    y_train = pd.read_csv('/Users/cp/Documents/dsi/capstone2/capstone2/data/train_setVIF_wednesday_target.csv')
    X_test = pd.read_csv('/Users/cp/Documents/dsi/capstone2/capstone2/data/validation_setVIF_wednesday_features.csv')
    y_test = pd.read_csv('/Users/cp/Documents/dsi/capstone2/capstone2/data/validation_setVIF_wednesday_target.csv')
  
    # X_train = pd.read_csv('/Users/cp/Documents/dsi/capstone2/capstone2/data/rna_jupyternotebook_df_wednesday.csv')
    # y_train = pd.read_csv('/Users/cp/Documents/dsi/capstone2/capstone2/data/target_death_short_col.csv')
    # X_test = pd.read_csv('/Users/cp/Documents/dsi/capstone2/capstone2/data/validation_X_set.csv')
    # y_test = pd.read_csv('/Users/cp/Documents/dsi/capstone2/capstone2/data/validation_y_set.csv')
    
    y_train = np.array(y_train).reshape(-1)
    y_test =  np.array(y_test).reshape(-1) 


    logistic2_regression_grid = {'C':[0.1 ,0.5, 0.8, 0.03065, 0.0307, 0.03075, 0.03077]
                       ,'solver':['liblinear']
                       ,'class_weight':['balanced']
                       ,'penalty':['l1']} 
    
    logistic_regressionCV_grid = {'Cs':[1,2,4,29]
                       ,'cv':[4]
                       ,'solver':['liblinear']
                       ,'class_weight':['balanced']
                       ,'penalty':['l1'] 
                        }

    random_forest_grid = {'max_depth': [2, 4, 8]
                     ,'max_features': ['sqrt', 'log2', None]
                     ,'min_samples_leaf': [1, 2, 4]
                     ,'min_samples_split': [2, 4]
                     ,'bootstrap': [True, False]
                     ,'class_weight': ['balanced']
                    ,'n_estimators': [5,10,25,50,100,200]
                     }
                    
    gradient_boosting_grid = {'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5]
                         ,'max_depth': [2, 4, 8]
                         ,'subsample': [0.25, 0.5, 0.75, 1.0]
                         ,'min_samples_leaf': [1, 2, 4]
                         ,'max_features': ['sqrt', 'log2', None]
                         ,'n_estimators': [5,10,25,50,100,200]
                         }

    
    results = optimize_model2_randomCV(LogisticRegression(), logistic2_regression_grid, X_train, y_train, scoring= 'roc_auc')
    

    best_model_predictor(results, X_test, y_test)