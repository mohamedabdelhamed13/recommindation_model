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


number_of_samples = 5000
df  = df().sample(number_of_samples)


# In[6]:


# encoding UserID and ProductID to simple integers to improve computation effeciency 
# Maintaing a map to get back the decoded UserID and ProductID after the calculations .

#add unique userID's to list
user_ids = df["UserID"].unique().tolist()
#encoding unique userID's 
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
#decoding unique userID's 
userencoded2user = {i: x for i, x in enumerate(user_ids)}

nurse_ids = df["nurseID"].unique().tolist()
nurse2nurse_encoded = {x: i for i, x in enumerate(nurse_ids)}
nurse_encoded2nurse = {i: x for i, x in enumerate(nurse_ids)}

#new cols to input simple integers form 
df["user"] = df["UserID"].map(user2user_encoded)
df["nurse"] = df["nurseID"].map(nurse2nurse_encoded)

num_users = len(user2user_encoded)
num_nurse = len(nurse_encoded2nurse)

df['Rating'] = df['Rating'].values.astype(np.float32)

min_rating = min(df['Rating'])
max_rating = max(df['Rating'])


# In[7]:


df = df.sample(frac=1, random_state=42)
x = df[["user", "nurse"]].values
#convert rating values to simple form (0 <= R <= 1)
y = df["Rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

train_indices = int(0.7 * df.shape[0])
val_indices = int(0.9 * df.shape[0]) 


x_train, x_val, x_test , y_train, y_val , y_test = (
    x[:train_indices],
    x[train_indices:val_indices],
    x[val_indices : ] , 
    y[:train_indices],
    y[train_indices:val_indices], 
    y[val_indices : ]
)


# #### Recommendor system using Matrix Factorisation using neural networks

# In[8]:


EMBEDDING_SIZE = 40

class Recommender(keras.Model):
    def __init__(self, num_users, num_nurse, embedding_size):
        super(Recommender, self).__init__()
        self.num_users = num_users
        self.num_nurse = num_nurse
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.nurse_embedding = layers.Embedding(
            num_nurse,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.nurse_bias = layers.Embedding(num_nurse, 1)
        
    def call(self, inputs):
        
        user_vector = self.user_embedding(inputs[:, 0])
        nurse_vector = self.nurse_embedding(inputs[:, 1])
        
        user_bias = self.user_bias(inputs[:, 0])
        nurse_bias = self.nurse_bias(inputs[:, 1])
        
        dot_prod = tf.tensordot(user_vector, nurse_vector, 2)

        x = dot_prod + user_bias + nurse_bias
        
        return tf.nn.sigmoid(x)
    
    def getRecomendation(self , df , user , k )  : 
    
        encoded_user = user2user_encoded[user]

        all_nurses = df['nurse'].unique() 
        nurses = df[df.user == encoded_user]['nurse'].values
        remainder = list(set(all_nurses) - set(nurses))
        n = len(remainder) 
        out = np.empty((n, 2),dtype=int)
        out[: ,  0 ] = encoded_user
        out[ : , 1 ] = remainder[:None]
        output = self.predict(out)

        ndx = map(lambda x : nurse_encoded2nurse[x] , remainder )
        vals = output[: , 0 ]

        return pd.Series(index = ndx , data = vals).sort_values(ascending = False )[ :k ].index
    


# In[9]:


model = Recommender(num_users, num_nurse, EMBEDDING_SIZE)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=0.001)
)


# In[10]:


history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=32,
    epochs=10,
#     verbose=1,
    validation_data=(x_val, y_val)
)


# In[19]:


import pickle

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)


# In[13]:


u = df['UserID'].sample(1).values[0]
K = 10 
top_10_nurse = model.getRecomendation(df , u ,K )

print("Top {k} recommendations for userID  : {user} are - {l} ".format( k = K  , user = u , l = list(top_10_nurse)))


# In[14]:


df[df['nurseID'].isin(top_10_nurse)]


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
    nurse_id =  flask.request.get_json()

    # Make a prediction
    prediction = model.getRecomendation(df() , nurse_id ,10 )

    # Return the prediction
    return print('  ')

# Run the API
if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




