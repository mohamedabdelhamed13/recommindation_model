#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from firebase_admin import credentials
from firebase_admin import db
import firebase_admin


# In[3]:


def df2():

    # json private key file from firebase concole
    cred = credentials.Certificate("C:\\Users\\moham\\Downloads\\nursing-project-5737d-firebase-adminsdk-cl1mq-17cbf7a29b.json")
    #URL for firebase
    firebase_admin.initialize_app(cred )
    #path for data
    ref = db.collection('Request')
    values = ref.get()
    #convert to dataframe 
    df = pd.DataFrame.from_dict(values)
    return df


# In[4]:


def df():
    data = pd.read_csv("ratings_nurses.csv" , header = None )
    data.columns = ["nurseID" , "UserID" , "Rating" , "Time" ] 
    data['Rating'] = data['Rating'].astype('int8')
    data.drop('Time' , axis = 1 , inplace = True )
    data = data.sort_values(by = "UserID") 
    return data


# In[5]:



df  = df()


# In[6]:




# In[1]:


import flask
import pickle


# Load the model from a file.
with open('model.pkl', 'rb') as f:
     model = pickle.load(f)
    
# Create the API
app = flask.Flask(__name__)

@app.route("/")
def predict():
    # Get the input data

    # Return the prediction
    return print('  ')

# Run the API
if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




