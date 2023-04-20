Amitesh Arya
CS 301
Milestone 4

Documentation:
  Code:
    The code is already documented decently with inline comments, however i will provide an overview documentation of each step below.
    For Milestone 2, I used a XGBoost Regression Model from the XGboost library and I trained it using the training data from kaggle. After importing the train.csv into a     dataframe, I seperated the last column, SalePrice, into a seperate dataframe. I fit the xgboost regression model onto these two dataframes and the only parameter i       gave the model was the learning rate, hence it performed pretty badly.
    For milestone 3, I reused my model from milestone 2 and I used optuna to find the best parameters for the xgboost model. 
    I ran 100 trials and gave it a handful of parameters and potential value ranges and let optuna decide on the best parameters. I then used my parameters from optuna and created the xgboost regression model and fit it on the same dataframes from milestone 2. This model performed a lot better due to the parameter tuning. I tested the accuracy of my model using the data from test.csv and found the absolute error between the predictions on test.csv and the real Sale prices from solution.csv.
    I used streamlit to setup the web app and added sliders for the most impactful features. I deployed the web app using streamlit cloud because I used the new cache  
    option in streamlit to make my application faster, however this cache option was not supported in Hugging Face.
  
  
  Results:
    The performance of the model was tested using absolute error and the unseen test.csv data and I found that the optuna tuning significantly improved the model   
    performance to be within roughly $15,000 of the actual sale price. From looking at the SHAP Summary plots, the most impactful feature was the Overall Quality feature,     the rest can be seen in the shap summary plot from the milestone2 notebook. 


A lot of the files are not relavant so please look only at optimizedModel.py and the streamlit cloud link. Thank You
Streamlit Cloud Link: https://amitesharya-milestone-1-optimizedmodel-milestone3-bx3em1.streamlit.app/
