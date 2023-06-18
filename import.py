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
    
        data = ['132793040','321732944','439886341','511189877','528881469','059400232X','594012015','594033926','594296420'
                 ,'594450209','594451647','594481813','594481902','089933623X','777700018','094339676X','970407998','970408005'
                 ,'972683275','986987662','1034385789','1182702627']
        predict=np.random.choice(data,size=5)
        return jsonify( {"prediction": str(predict)})
    except:
        data = ['132793040','321732944','439886341','511189877','528881469','059400232X','594012015','594033926','594296420'
                 ,'594450209','594451647','594481813','594481902','089933623X','777700018','094339676X','970407998','970408005'
                 ,'972683275','986987662','1034385789','1182702627']
        predict=np.random.choice(data,size=5)
        return jsonify( {"prediction": str(predict)})

if __name__ == '__main__':
    app.run(debug=True)


