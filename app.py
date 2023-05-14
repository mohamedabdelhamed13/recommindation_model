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



from flask import Flask

app = Flask(__name__)

@app.route("/")
def reco():
    return "<p>     </p>"






