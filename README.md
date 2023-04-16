Amitesh Arya
CS 301

Completed XGBoost Model (optimizedModel.py) and deployed to streamlit cloud. I ran optuna for 100 trials to find the best parameters and then i hardcoded them 
into the model, so it would not need to be run again. In order to speed up the web ui, I cached the model generation so it only needs to be run once, however
the reccomended form of caching by streamlit, is not supported by Hugging Face, I believe this is because Hugging Face uses an older version of streamlit. Instead,
I deployed directly through streamlit cloud which updates according to this branch of this repository. 

A lot of the files are not relavant so please look only at optimizedModel.py and the streamlit cloud link. Thank You
Streamlit Cloud Link: https://amitesharya-milestone-1-optimizedmodel-milestone3-bx3em1.streamlit.app/
