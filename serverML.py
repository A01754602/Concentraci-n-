#Python Libraries 
from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
#Files management
import os 
from werkzeug.utils import secure_filename

#Load model
dt = joblib.load('dt1_ml.joblib')
#Create Flask App
server = Flask(__name__)

#Define route
@server.route('/predictjson', methods=['POST'])
def predictjson():
    #Procesar datos de entrada
    data = request.json
    print(data)
    input_data = np.array([
        data['pH'],
        data['sulphates'],
        data['alcohol']
    ])
    #Predecir utilizando la entrada y el modelo
    result = dt.predict(input_data.reshape(1,-1))
    #Enviar respuesta
    return jsonify({'Prediction': str(result[0])})
if __name__ == '__main__':
    server.run(debug=True, host='0.0.0.0', port=8080)