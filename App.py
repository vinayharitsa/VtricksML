from flask import Flask,render_template,url_for,request
import pandas as pd 
from sklearn.externals import joblib
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    model = open("Heart_model.pkl","rb")
    clfr = joblib.load(model)

    if request.method == 'POST':
        params = input_params()
        ip_features = np.asarray(params).reshape(1,-1)
        my_prediction = clfr.predict(ip_features)
        details = request.form
        name = details['name']
        age = details['age']
        sex = details['sex']
        pain = details['cp']
        rBp = details['trestbps']
        cholestrol = details['chol']
        fBp = details['fbs']
        ecg = details['restecg']
        hRate = details['thalach']
        angina = details['exang']
        depression = details['oldpeak']
        peak = details['slope']
        vessels = details['ca']
        thal = details['thal']
        
        return render_template('predict.html',prediction = int(my_prediction[0]))

def input_params():
    params = []
    #params.append(request.form('name'))
    params.append(request.form['age'])
    params.append(request.form['sex'])
    params.append(request.form['cp'])
    params.append(request.form['trestbps'])
    params.append(request.form['chol'])
    params.append(request.form['fbs'])
    params.append(request.form['restecg'])
    params.append(request.form['thalach'])
    params.append(request.form['exang'])
    params.append(request.form['oldpeak'])
    params.append(request.form['slope'])
    params.append(request.form['ca'])
    params.append(request.form['thal'])
    return params

if __name__ == '__main__':
    app.run(debug=True)