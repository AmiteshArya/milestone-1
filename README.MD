Milestone 2. 
Please run notebook in nbviewer to see the javascript graphs from shap. 
https://nbviewer.org./github/AmiteshArya/milestone-1/blob/milestone2/milestone2.ipynb

Please see embedded notebook comments for thought process and procedure.

Additional Comments:

From looking at the SHAP values in each of the first three graphs, I can see that the two most impactful features were the 'OverallQual' and the 'GrLivArea', which makes 
sense since they each tell a lot about a home. 

Looking at the interaction SHAP values, some notable correlations are the year built and overall condition, with the overall condition being rated higher on newer houses. 
There are also more notable correlations that make sense when looking at the plots.

Note: I only used the 37 numerical features in my model and omitted the 43 non numerical features. I was worried about doing this however the model still performed decently well.
It was within a 15% error on the test data using just the 37 numerical features. 

I was not able to figure out how to add the non-numerical features into the model, I belive that the xgboosting does not currently support categorial features using the 
gradient boosting tree model. 
