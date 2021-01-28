# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 20:48:04 2020

@author: hp
"""


# importing pandas library 
import pandas as pd 

# reading the given csv file 
# and creating dataframe 
account = pd.read_csv("comments79.json",header=None,delimiter = ',')

abc=account.T 

# store dataframe into csv file 
abc.to_csv('comments79.csv', index = None) 
