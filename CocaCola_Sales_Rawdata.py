# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 22:28:03 2021

@author: DELL
"""

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading the dataset
cocacola=pd.read_excel('CocaCola_Sales_Rawdata.xlsx')
quarters=['Q1','Q2','Q3','Q4']
quarters=pd.DataFrame(quarters)
Quarters=pd.DataFrame(np.tile(quarters,(11,1)))
Cocacola=pd.concat([cocacola,Quarters],axis=1)
Cocacola=Cocacola.dropna()
Cocacola.columns=['Quarter','Sales','quarters']

#Creating the dummies
Quarter_dummies=pd.get_dummies(Cocacola['quarters'])
Cocacola=pd.concat([Cocacola,Quarter_dummies],axis=1)
Cocacola['t']=np.arange(1,43)
Cocacola['t_sq']=Cocacola['t']*Cocacola['t']
Cocacola['log_Sales']=np.log(Cocacola['Sales'])

#Splitting the Train and Test
Train=Cocacola[0:30]
Test=Cocacola[30:]

#plot
plt.plot(Cocacola.iloc[:,1])
Test.set_index(np.arange(1,13),inplace=True)

######### Linear ###########
import statsmodels.formula.api as smf
lin_model=smf.ols('Sales~t',data=Train).fit()
pred_lin=lin_model.predict(Test['t'])
error_lin=Test['Sales']-pred_lin
rmse_lin=np.sqrt(np.mean(error_lin**2))
#rmse_lin=714.01

########## Exponential ########
import statsmodels.formula.api as smf
exp_model=smf.ols('log_Sales~t',data=Train).fit()
pred_exp=exp_model.predict(Test['t'])
error_exp=Test['Sales']-pred_exp
rmse_exp=np.sqrt(np.mean(error_exp**2))
#rmse_exp=4252.188

######### Quadratic ##########
import statsmodels.formula.api as smf
quad_model=smf.ols('Sales~t+t_sq',data=Train).fit()
pred_quad=quad_model.predict(Test[['t','t_sq']])
error_quad=Test['Sales']-pred_quad
rmse_quad=np.sqrt(np.mean(error_quad**2))
#rmse_quad=646.27

######### Additive Seasonality ########
import statsmodels.formula.api as smf
add_sea_model=smf.ols('Sales~Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_sea_model=add_sea_model.predict(Test[['Q1','Q2','Q3','Q4']])
error_add_sea=Test['Sales']-pred_add_sea_model
rmse_add_sea=np.sqrt(np.mean(error_add_sea**2))
#rmse_add_sea=1778.006

####### Additive Seasonality Quadratic ##########
import statsmodels.formula.api as smf
add_sea_quad=smf.ols('Sales~Q1+Q2+Q3+Q4+t+t_sq',data=Train).fit()
pred_add_sea_quad=add_sea_quad.predict(Test[['Q1','Q2','Q3','Q4','t','t_sq']])
error_add_sea_quad=Test['Sales']-pred_add_sea_quad
rmse_add_sea_quad=np.sqrt(np.mean(error_add_sea_quad**2))
#rmse_add_sea_quad=586.05

######### Multiplicative Seasonality ##########
import statsmodels.formula.api as smf
mul_sea_model=smf.ols('log_Sales~Q1+Q2+Q3+Q4',data=Train).fit()
pred_mul_sea=mul_sea_model.predict(Test[['Q1','Q2','Q3','Q4']])
error_mul_sea=Test['Sales']-pred_mul_sea
rmse_mul_sea=np.sqrt(np.mean(error_mul_sea**2))
#rmse_mul_sea=4252.638

######## Multiplicative Additive Seasonality #############
import statsmodels.formula.api as smf
mul_add_sea=smf.ols('log_Sales~t+Q1+Q2+Q3+Q4',data=Train).fit()
pred_mul_add_sea=mul_add_sea.predict(Test[['t','Q1','Q2','Q3','Q4']])
error_mul_add_sea=Test['Sales']-pred_mul_add_sea
rmse_mul_add_sea=np.sqrt(np.mean(error_mul_add_sea**2))
#rmse_mul_add_sea=4252.185

data={'model':['lin_model','exp_model','quad_model','add_sea_model','add_sea_quad','mul_sea_model','mul_add_sea'],
      'rmse_val':[rmse_lin,rmse_exp,rmse_quad,rmse_add_sea,rmse_add_sea_quad,rmse_mul_sea,rmse_mul_add_sea]}
rmse_table=pd.DataFrame(data)
rmse_table

#Additive Seasonality Quadratic is having the least rmse value
#So, Additive Seasonality Quadratic is the best model



























