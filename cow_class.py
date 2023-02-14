#!/usr/bin/env python
# coding: utf-8

# ### Import libraries

# In[60]:


import os
import glob
import pandas as pd
import numpy  as np
from ipywidgets import interact
import scipy
from scipy import stats
import scipy.linalg
import pywt
import pickle
import sys
from numpy.linalg import norm


# ### Create BaseClass

# In[62]:


class CowSignalsProcessing:
    
    def make_diagnosis(self, timeseries):
        try:
            x = self.find_nan(timeseries) 
        except :
            print("Problems with my find_nan function")
            print("Unexpected error:",sys.exc_info())
            return -1
            
        try:
            y = self.missing_val(x)
        except :
            print("Problems with my missing_val function")
            print("Unexpected error:",sys.exc_info())
            return -1
            
        try:
            z = self.splitter_4(y)
        except :
            print("Problems with my splitter_4 function")
            print("Unexpected error:",sys.exc_info())
            return -1
            
        try:
            h = self.calc_feature_vector(y)
            
        except :
            print("Problems with my calc_feature_vector function")
            print("Unexpected error:",sys.exc_info())
            return -1
        
        try:
            
            result = self.class_model.predict(np.expand_dims(h, axis = 0))
            print("Prediction done")
            return result
        except :
            print("Problems with Predict function")
            print("Unexpected error:", sys.exc_info())
            return -1
    
    def load_classification_model(self, cl_model_location):
        try:
            self.class_model = pickle.load(open(cl_model_location, 'rb'))
            print("Classification model loaded.")
        except:
            print("Can't load classification model")
            print("Unexpected error:", sys.exc_info())
        
    def find_nan(self,x):
        if x['Value'].all() == '4080':
            val = x['Value'].replace('4080', np.NaN)
            return val
        else:
            return x 
    
    def missing_val(self,data):
        x_rep = data.interpolate(method='linear')
        return x_rep

    def splitter_4(self, lst):
        lst1= []
        a = lst[:83328]
        b = lst[83328:166656]
        c = lst[166656:249984]
        d = lst[249984:333312]

        a1 = a.reset_index(drop = True)
        b1 = b.reset_index(drop = True)
        c1 = c.reset_index(drop = True)
        d1 = d.reset_index(drop = True)

        data_splitted = pd.concat([a1,b1,c1,d1],axis = 1)
        data_splitted.columns= ['Value 1 ','Value 2 ','Value 3 ','Value 4 ']
        lst1.append(data_splitted)
        return lst1

    
    def feature_mean(self, matrix):
        res = self.splitter_4(matrix)
        ret = np.mean(res[0], axis = 0).values.flatten()
        return ret

    def feature_stddev(self, matrix):
        res = self.splitter_4(matrix)
        ret = np.std(res[0], axis = 0, ddof = 1).values.flatten()
        return ret
    
    def feature_moments(self, matrix):
        res = self.splitter_4(matrix)
        skw = scipy.stats.skew(res[0], axis = 0, bias = False)
        krt = scipy.stats.kurtosis(res[0], axis = 0, bias = False)
        ret  = np.append(skw, krt)
        return ret
    
    def feature_max(self, matrix):
        res = self.splitter_4(matrix)
        ret = np.max(res[0], axis = 0).values.flatten()
        return ret
    
    def feature_min(self, matrix):
        res = self.splitter_4(matrix)
        ret = np.min(res[0], axis = 0).values.flatten()
        return ret
    
    def feature_covariance_matrix(self,matrix):
        res = self.splitter_4(matrix)
        covM = np.cov(res[0].T)
        indx = np.triu_indices(covM.shape[0])
        ret  = covM[indx]
        return ret, covM
    
    def DWT(self, matrix):
        res = self.splitter_4(matrix)
        resp = pywt.dwt(res[0], 'db4',axis = 0)
        L2_norm=norm(resp)
        return L2_norm
    
    def calc_feature_vector(self, matrix):
        x = self.feature_mean(matrix)
        var_values = x

        x = self.feature_stddev(matrix)
        var_values = np.hstack([var_values, x])

        x = self.feature_moments(matrix)
        var_values = np.hstack([var_values, x])

        x = self.feature_max(matrix)
        var_values = np.hstack([var_values, x])
    

        x = self.feature_min(matrix)
        var_values = np.hstack([var_values, x])


        x,covM = self.feature_covariance_matrix(matrix)
        var_values = np.hstack([var_values, x])
    
        x = self.DWT(matrix)
        var_values = np.hstack([var_values, x])

        return var_values
    

