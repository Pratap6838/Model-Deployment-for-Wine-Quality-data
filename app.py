import numpy as np
from flask import Flask,render_template,request
import pickle
#from sklearn.ensemble import RandomForestRegressor


app = Flask(__name__)
rfr_model = pickle.load(open('model.pkl','rb'))

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():
    
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    prediction = rfr_model.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('index.html',prediction_text='Quality of Wine : {}'.format(output))




if __name__ == '__main__':
    app.run(debug=True, port=8080)