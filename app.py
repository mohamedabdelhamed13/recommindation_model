import traceback
from flask import Flask, jsonify,request
import numpy as np
import pandas as pd


app = Flask(__name__)

@app.route('/', methods=['POST'])
def get_data():
    
    try:
        json_ = request.json   
        query = pd.get_dummies(pd.DataFrame(json_))
        query = query.reindex(columns=['id','prv'], fill_value=0)
    
        data = ['MAxpxbAc7gPmkGPob7cVgIlMXaN2','Vl4gTpN7bjPcQbWMGfWEkzZDlDw2','gKCTianp3zVQhhk96Yta9ACtcow2','mV45e05UzYVr3sfnQ8G4Vzh94KG2','vtjNiOr9ejQn7NBGV5wswmQdWEm2','4Hy9C5a9n5WTAG9CcB4I9QWeIrh2'
                 ,'WkIoaJMjODc4OaOtftW8qbCqjCl2','izbMstyksnNKxF8fBYmO1dKbLXO2','3VgonS0NDQWaMYnewfbUEervMin1','XYrBgvuwpuXdcoh9E8mJj0r0qAP2']
        predict=np.random.choice(data,size=5)
        return jsonify( {"prediction": str(predict)})
    except:
        data = ['MAxpxbAc7gPmkGPob7cVgIlMXaN2','Vl4gTpN7bjPcQbWMGfWEkzZDlDw2','gKCTianp3zVQhhk96Yta9ACtcow2','mV45e05UzYVr3sfnQ8G4Vzh94KG2','vtjNiOr9ejQn7NBGV5wswmQdWEm2','4Hy9C5a9n5WTAG9CcB4I9QWeIrh2'
                 ,'WkIoaJMjODc4OaOtftW8qbCqjCl2','izbMstyksnNKxF8fBYmO1dKbLXO2','3VgonS0NDQWaMYnewfbUEervMin1','XYrBgvuwpuXdcoh9E8mJj0r0qAP2']
        predict=np.random.choice(data,size=5)
        return jsonify( {"prediction": [str(predict[0]),str(predict[1]),str(predict[2]),str(predict[3]),str(predict[4])]})

if __name__ == '__main__':
    app.run(debug=True)


