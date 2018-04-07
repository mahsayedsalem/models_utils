# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 21:36:27 2018

@author: mahsayedsalem
"""

import pandas as pd
import warnings 
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)

def plot_using_pandas(kind, x_feature, y_feature, df):
    
    df.plot(kind=kind, x=x_feature, y=y_feature)
 

def plot_scatter_using_sns(x_feature, y_feature, labels, s=5, df):
    
    sns.FacetGrid(df, hue=labels, size=s) \
   .map(plt.scatter, x_feature, y_feature) \
   .add_legend()


def plot_boxplot_using_sns(y_feature, labels, df):
    
    ax = sns.boxplot(x=labels, y=y_feature, data=df)
    ax = sns.stripplot(x=labels, y=y_feature, data=df, jitter=True, edgecolor="gray")
    

def plot_violinplot_using_sns(y_feature, labels, df, s=6):
    
    sns.violinplot(x=labels, y=y_feature, data=df, size=s)
    

def plot_kdeplot_using_sns(x_feature, labels, df, s=6):
    
    sns.FacetGrid(df, hue=labels, size=s) \
   .map(sns.kdeplot, x_feature) \
   .add_legend()
   
   
def plot_pairplot_using_sns(labels, df, s=3):
    
    sns.pairplot(df, hue=labels, size=s)
    

def plot_histogram_using_sns(df, x_feature):
    
    sns.distplot(df[x_feature])
    

def plot_heatmap_using_sns(df):
    
    corrmat = df.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    

def plot_heatmap_filtered_using_sns(df, k, labels):
   
    cols = corrmat.nlargest(k, labels)[labels].index
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': k}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()