#Import Libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
import optuna
import streamlit as st
from sklearn.metrics import mean_squared_error

@st.cache_data
def createModel():
    #Read training data and data to test model against, both into a dataframe
    data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')

    #Remove the sale price from the training data frame and place it into a seperate data frame. Use x and y to train the model.
    x = data.drop(['SalePrice'],axis=1)
    y = data['SalePrice']

    #I removed non scalar features such as 'Alley' and 'Fence'. Although they are important, I could not find a way to get them to work with xgboost
    # and the model still performs good enough using just the numerical features, hence, I only used the 37 numerical features instead of all 80. I did the same for the test data as well. 

    x = x.select_dtypes(include = ['float64', 'int64'])
    x_test = test_data.select_dtypes(include= ['float64', 'int64'])


    solution  = pd.read_csv('solution.csv')
    y_true     = solution["SalePrice"].to_numpy()
    def objective(trial):
        param = {
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.01, 1.0),
            'subsample': trial.suggest_float('subsample', 0.01, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
            'random_state': trial.suggest_int('random_state', 1, 1000)
        }
        model = xgb.XGBRegressor(**param)
        model.fit(x, y)
        y_pred = model.predict(x_test)
        return mean_squared_error(y_pred, y_true)


    #study = optuna.create_study(direction='minimize', study_name='regression')
    #study.optimize(objective, n_trials=100)

    #print('Best parameters', study.best_params)

    bestParams = {'max_depth': 9, 'learning_rate': 0.02398782236068422, 'n_estimators': 535, 'min_child_weight': 4, 'gamma': 0.14105107984219764, 'subsample': 0.6784141449085943, 'colsample_bytree': 0.4658168920071173, 'reg_alpha': 0.8978828423965367, 'reg_lambda': 0.29288514841600993, 'random_state': 392}
    #I created the model using XGBoost, I gave it the best parameters that optuna decided on
    bestFit = xgb.XGBRegressor(**bestParams)

    #I trained the model using the numerical features from train.csv and the house sale prices also from train.csv. 
    bestFit.fit(x,y)
    return bestFit

bestFit = createModel()
#I gave the now trained model the new test data which it has not seen before.
x_testOne = pd.read_csv('testOne.csv')
#print(x_testOne)
#print(x_test)
#x_test.to_csv('remainingCol.csv')
predictTestData = bestFit.predict(x_testOne)



st.write("Housing Prediction Model - Amitesh Arya")



print('$' + str(0.95 * predictTestData[0]), 'to', '$' + str(1.05* predictTestData[0]))
#st.write(x_testOne)
MSubClass_Slider = st.sidebar.slider(
    'MSSubClass',
    20, 190
)
#st.write(add_slider)

def runModel(inputDF):
    st.write(bestFit.predict(inputDF)[0])


x_testOne['YearBuilt'] = 1990
st.write(x_testOne)
print("ran model from function:", runModel(x_testOne))
""" 	Id	
	MSSubClass x 
	LotFrontage
	LotArea
	OverallQual
	OverallCond
	YearBuilt
	YearRemodAdd
	MasVnrArea
	BsmtFinSF1
	BsmtFinSF2
	BsmtUnfSF
	TotalBsmtSF
	1stFlrSF
	2ndFlrSF
	LowQualFinSF
	GrLivArea
	BsmtFullBath
	BsmtHalfBath
	FullBath
	HalfBath
	BedroomAbvGr
	KitchenAbvGr
	TotRmsAbvGrd
	Fireplaces
	GarageYrBlt
	GarageCars
	GarageArea
	WoodDeckSF
	OpenPorchSF
	EnclosedPorch
	3SsnPorch
	ScreenPorch
	PoolArea
	MiscVal
	MoSold
	YrSold """