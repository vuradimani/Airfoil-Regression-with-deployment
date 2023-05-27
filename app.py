import pickle
import numpy as np
import pandas as pd
import flask
from flask import jsonify
from flask import Flask, request, app, url_for, render_template
from sklearn.linear_model import *
from sklearn.ensemble import *
from flask import Response
from flask_cors import CORS

app=Flask(__name__)

#load the pickle file
model=pickle.load(open('model1.pkl','rb'))

#create api for predict
@app.route('/predict_api',methods=['POST'])
def predict_api():

    data=request.json['data']
    print(data)
    new_data =[list(data.values())]
    output= model.predict(new_data)[0]
    return jsonify(output)

@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    data=[float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    print(data)

    output= model.predict(final_features)[0]
    print(output)

    return render_template('home.html', prediction_text= "Airfoil pressure is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)
