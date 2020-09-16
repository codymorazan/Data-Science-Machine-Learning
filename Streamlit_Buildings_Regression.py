# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 16:15:45 2020

@author: Corndog Expert
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error


buildings = pd.read_csv("C:\\Users\\Corndog Expert\\Documents\\Python Scripts\\Personal Sets and Work\\building_dataset_upload.csv")

#Quick visual of first rows
print(buildings.head(10))
print(buildings.columns)


#Drop empty columns as well as other unwanted dependent variable
buildings = buildings.drop(['Heating Load','Unnamed: 10', 'Unnamed: 11'], axis=1)
print(buildings.count())

print(buildings.isnull().any())
print(buildings.isnull().sum())
buildings = buildings.fillna(method='ffill')
print(buildings.isnull().any())
print(buildings.isnull().sum())


#Quick inspection of summary stats
print(buildings.describe())


#Some initial visualizations
fig, ax = plt.subplots()
ax.plot(buildings['Surface Area'], buildings['Cooling Load'])
ax.plot(buildings['Wall Area'], buildings['Cooling Load'])
ax.plot(buildings['Roof Area'], buildings['Cooling Load'])
plt.show()

fig, ax = plt.subplots()
num_bins = 20
n, bins, patches = ax.hist(buildings['Roof Area'], num_bins, density=True)
fig.tight_layout()
plt.show()

fig, ax = plt.subplots()
num_bins = 20
n, bins, patches = ax.hist(buildings['Surface Area'], num_bins, density=True)
fig.tight_layout()
plt.show()

fig, ax = plt.subplots()
num_bins = 20
n, bins, patches = ax.hist(buildings['Wall Area'], num_bins, density=True)
fig.tight_layout()
plt.show()


#Let's split into train and test data
X = buildings.drop(['Cooling Load'], axis=1)
y = buildings['Cooling Load']


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=27)


#Let's normalize and standardize
norm = MinMaxScaler().fit(X_train)
X_train_norm = norm.transform(X_train)
X_test_norm = norm.transform(X_test)

X_train_stand = X_train.copy()
X_test_stand = X_test.copy()
scaler = StandardScaler().fit(X_train_stand)
X_train_stand = scaler.transform(X_train_stand)
X_test_stand = scaler.transform(X_test_stand)


#Let's fit a multiple linear regression model
lr = linear_model.LinearRegression()
training_X = [X_train, X_train_norm, X_train_stand]
test_X = [X_test, X_test_norm, X_test_stand]
index=['Original','Normalized','Standardized']
mse_list = []

for i in range(len(training_X)):
    lr.fit(training_X[i], y_train)  #fit on X_train and y_train
    print(index[i])
    print("Intercept: " + str(lr.intercept_))
    print("Coefficients" + str(lr.coef_))
    print()
    prediction = lr.predict(test_X[i])   #predict for X_test
    mse_list.append(mean_squared_error(y_test, prediction))  #compare y_test and prediction

for i in range(len(index)):
    print(index[i] + ": " + str(mse_list[i]))


#Choose best performing data
lr_final = linear_model.LinearRegression()
lr_final.fit(training_X[0], y_train)
print(lr_final.coef_, lr_final.intercept_)


#Make prediction using chosen data

st.title("Cooling Load Regression Predictor")
st.write("""Multiple Linear Regression""")
st.text(" ")
st.write("""This app predicts required cooling loads""")

st.sidebar.header("Choose your parameters")

def input_parameters():
    rc = st.sidebar.slider('Relative Compactness', min_value=0.0, max_value=1.0, value=0.68, step=0.01)
    sa = st.sidebar.slider('Surface Area', min_value=500, max_value=1000, value=642, step=1)
    wa = st.sidebar.slider('Wall Area', min_value=200, max_value=500, value=298, step=1)
    ra = st.sidebar.slider('Roof Area', min_value=150, max_value=400, value=300, step=1)
    oh = st.sidebar.slider('Overall Height', min_value=3.0, max_value=8.0, value=4.3, step=0.1)
    o = st.sidebar.slider('Orientation', min_value=1.0, max_value=5.0, value=2.22, step=0.01)
    ga = st.sidebar.slider('Glazing Area', min_value=12.0, max_value=30.0, value=17.18, step=0.01)
    gad = st.sidebar.slider('Glazing Area Distribution', min_value=10, max_value=50, value=42, step=1)

    features = [rc, sa, wa, ra, oh, o, ga, gad]
    return features

df = input_parameters()


predict_value = lr_final.intercept_
for i in range(len(lr_final.coef_)):
    predict_value += (df[i]*lr_final.coef_[i])
print(predict_value)

st.subheader("Prediction")
st.write(predict_value)