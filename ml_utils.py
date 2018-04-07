# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 22:41:48 2018

@author: mahsayedsalem
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings 
warnings.filterwarnings("ignore")

def accuracy_report(y_test, y_pred):
    
    cm = confusion_matrix(y_test,y_pred)
    print('Classification report: \n',classification_report(y_test,y_pred))
    sns.heatmap(cm,annot=True,fmt="d") 
    plt.show()


def KNN(x, y, percentage, k):
    
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = percentage)
    model = KNeighborsClassifier(n_neighbors = k)
    model.fit(x_train,y_train)
    filename = 'KNN_model_K_'+str(k)+'.sav'
    joblib.dump(model, filename)
    acc_train = round(model.score(x_train, y_train) * 100, 2)
    acc_test = round(model.score(x_test, y_test) * 100, 2)
    y_pred = model.predict(x_test)
    print('With KNN (k =',k,') train accuracy is: ', acc_train)
    print('With KNN (k =',k,') test accuracy is: ', acc_test)
    accuracy_report(y_test, y_pred)
    #loaded_model = joblib.load(filename)
    print('______________')
    return acc_train, acc_test
    

def ml_KNN_tuning(x, y, percentage):
    
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = percentage)
    neig = np.arange(1, 25)
    train_accuracy = []
    test_accuracy = []
    
    for i, k in enumerate(neig):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train,y_train)
        train_accuracy.append(knn.score(x_train, y_train))
        filename = 'Knn_tuning_model_K_'+str(k)+'.sav'
        joblib.dump(knn, filename)
        test_accuracy.append(knn.score(x_test, y_test))
    plt.figure(figsize=[13,8])
    plt.plot(neig, test_accuracy, label = 'Testing Accuracy')
    plt.plot(neig, train_accuracy, label = 'Training Accuracy')
    plt.legend()
    plt.title('k value VS Accuracy')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.xticks(neig)
    plt.show()
    print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))
    print('______________')
    return 1+test_accuracy.index(np.max(test_accuracy))


def logistic_regression(x, y, percentage):
    
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = percentage)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    acc_train = round(model.score(x_train, y_train) * 100, 2)
    acc_test = round(model.score(x_test, y_test) * 100, 2)
    y_pred = model.predict(x_test)
    filename = 'Logistic_Regression_model.sav'
    joblib.dump(model, filename)
    print('With logistic regression, train accuracy is: ',acc_train)
    print('With logistic regression, test accuracy is: ',acc_test)
    accuracy_report(y_test, y_pred)
    print('______________')
    return acc_train, acc_test
    
    
def guassian_naive_bayes(x, y, percentage):
    
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = percentage)
    model = GaussianNB()
    model.fit(x_train, y_train)
    acc_train = round(model.score(x_train, y_train) * 100, 2)
    acc_test = round(model.score(x_test, y_test) * 100, 2)
    y_pred = model.predict(x_test)
    filename = 'Guassian_Naive_Bayes_model.sav'
    joblib.dump(model, filename)
    print('With guassian naive bayes, train accuracy is: ',acc_train)
    print('With guassian naive bayes, test accuracy is: ',acc_test)
    accuracy_report(y_test, y_pred)
    print('______________')
    return acc_train, acc_test


def perceptron(x, y, percentage):
    
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = percentage)
    model = Perceptron()
    model.fit(x_train, y_train)
    acc_train = round(model.score(x_train, y_train) * 100, 2)
    acc_test = round(model.score(x_test, y_test) * 100, 2)
    y_pred = model.predict(x_test)
    filename = 'Perceptron_model.sav'
    joblib.dump(model, filename)
    print('With perceptron, train accuracy is: ', acc_train)
    print('With perceptron, test accuracy is: ', acc_test)
    accuracy_report(y_test, y_pred)
    print('______________')
    return acc_train, acc_test


def linear_svc(x, y, percentage):
    
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = percentage)
    model = LinearSVC() 
    model.fit(x_train, y_train)
    acc_train = round(model.score(x_train, y_train) * 100, 2)
    acc_test = round(model.score(x_test, y_test) * 100, 2)
    y_pred = model.predict(x_test)
    filename = 'Linear_SVC_model.sav'
    joblib.dump(model, filename)
    print('With linear svc, train accuracy is: ', acc_train)
    print('With linear svc, test accuracy is: ', acc_test)
    accuracy_report(y_test, y_pred)
    print('______________')
    return acc_train, acc_test


def SGD(x, y, percentage):
    
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = percentage)
    model = SGDClassifier()
    model.fit(x_train, y_train)
    acc_train = round(model.score(x_train, y_train) * 100, 2)
    acc_test = round(model.score(x_test, y_test) * 100, 2)
    y_pred = model.predict(x_test)
    filename = 'SGD_model.sav'
    joblib.dump(model, filename)
    print('With stochastic gradient decent, train accuracy is: ', acc_train)
    print('With stochastic gradient decent, test accuracy is: ', acc_test)
    accuracy_report(y_test, y_pred)
    print('______________')
    return acc_train, acc_test
    
    
def decision_tree(x, y, percentage):
    
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = percentage)
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    acc_train = round(model.score(x_train, y_train) * 100, 2)
    acc_test = round(model.score(x_test, y_test) * 100, 2)
    y_pred = model.predict(x_test)
    filename = 'Decision_Tree_model.sav'
    joblib.dump(model, filename)
    print('With decision tree, train accuracy is: ', acc_train)
    print('With decision tree, test accuracy is: ', acc_test)
    accuracy_report(y_test, y_pred)
    print('______________')
    return acc_train, acc_test


def random_forest(x, y, percentage, n):
    
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = percentage)
    model = RandomForestClassifier(n_estimators=n)
    model.fit(x_train, y_train)
    acc_train = round(model.score(x_train, y_train) * 100, 2)
    acc_test = round(model.score(x_test, y_test) * 100, 2)
    y_pred = model.predict(x_test)
    filename = 'Random_Forest_model.sav'
    joblib.dump(model, filename)
    print('With random forest, train accuracy is: ', acc_train)
    print('With random forest, test accuracy is: ', acc_test)
    accuracy_report(y_test, y_pred)
    print('______________')
    return acc_train, acc_test


def svm(x, y, percentage):
    
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = percentage)
    model = SVC()
    model.fit(x_train, y_train)
    acc_train = round(model.score(x_train, y_train) * 100, 2)
    acc_test = round(model.score(x_test, y_test) * 100, 2)
    y_pred = model.predict(x_test)
    filename = 'Support_Vector_Machine_model.sav'
    joblib.dump(model, filename)
    print('With support vector machines, train accuracy is: ', acc_train)
    print('With support vector machines, test accuracy is: ', acc_test)
    accuracy_report(y_test, y_pred)
    print('______________')
    return acc_train, acc_test


def full_test(x, y, percentage, k, n):
    
    acc_svc_train, acc_svc_test = svm(x, y, percentage)
    acc_random_forest_train, acc_random_forest_test = random_forest(x, y, percentage, n)
    acc_knn_train, acc_knn_test = KNN(x, y, percentage, k)
    acc_decision_tree_train, acc_decision_tree_test = decision_tree(x, y, percentage)
    acc_SGD_train, acc_SGD_test = SGD(x, y, percentage)
    acc_linear_svc_train, acc_linear_svc_test = linear_svc(x, y, percentage)
    acc_perceptron_train, acc_perceptron_test = perceptron(x, y, percentage)
    acc_guassian_naive_bayes_train, acc_guassian_naive_bayes_test = guassian_naive_bayes(x, y, percentage)
    acc_logistic_regression_train, acc_logistic_regression_test = logistic_regression(x, y, percentage)
    
    models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score_train': [acc_svc_train, acc_knn_train, acc_logistic_regression_train, 
              acc_random_forest_train, acc_guassian_naive_bayes_train, acc_perceptron_train, 
              acc_SGD_train, acc_linear_svc_train, acc_decision_tree_train],
    'Score_test': [acc_svc_test, acc_knn_test, acc_logistic_regression_test, 
              acc_random_forest_test, acc_guassian_naive_bayes_test, acc_perceptron_test, 
              acc_SGD_test, acc_linear_svc_test, acc_decision_tree_test]})

    model_sorted = models.sort_values(by='Score_test', ascending=False)
    print(model_sorted)