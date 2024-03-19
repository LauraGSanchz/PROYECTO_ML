"""""archivo donde corre la app"""
from flask import Flask, request, render_template,jsonify
import pickle
import os
import seaborn as sns
import pandas as pd


app = Flask(__name__)

# Ruta al archivo del modelo
modelo_ruta = 'C:\\Users\\laura\\OneDrive\\Desktop\\PROYECTO ML\\ml_project\\src\\models\\ignore\\forecaster_model.pkl'

# Verificar si el archivo existe antes de cargarlo
if os.path.exists(modelo_ruta):
    with open(modelo_ruta, 'rb') as archivo_modelo:
        modelo = pickle.load(archivo_modelo)
else:
    print("El archivo del modelo no existe en la ruta especificada.")

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_form')
def prediction_form():
    return render_template('predict.html')



@app.route('/predict', methods=['POST'])
def predict():
    steps = request.form.get('steps')  # Obtener el número de pasos del formulario
    
    if steps is None:
        return jsonify({'error': 'No se proporcionó el número de pasos.'}), 400
    
    # Convertir los pasos a un entero
    steps = int(steps)
    
    # Ahora, utiliza el número de pasos recibido para realizar la predicción
    predictions = modelo.predict(steps=steps)
    
    # Devolver los resultados como una respuesta JSON
    return jsonify({'predictions': list(predictions)})



# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     steps = int(data.get('steps'))  # Suponiendo que recibes el número de pasos como 'steps' en los datos JSON
    
#     if steps is None:
#         return jsonify({'error': 'No se proporcionó el número de pasos.'}), 400
    
#     # Ahora, utiliza el número de pasos recibido para realizar la predicción
#     predictions = modelo.predict(steps=steps)
    
#     return print(pd.DataFrame(predictions))




if __name__ == '__main__':
    app.run(port=8000, debug=True) #debug false hace que no se actualice la app y no se vean los cambos automaticamente en la web. No se puede dejar en true cuando subo la app. 


# @app.route('/datos/<name>', methods=['GET'])
# def user(name):
#     return render_template('datos.html', name=name)



# @app.route("/datos/<name>/<int:ind>") #ruta dinamica coge lo que ponemos en name y lo muestra en la web 
# def funcion(name, ind):
#     mylist = ['elemento', 'elemento1', 'elemento2','elemento3']
#     mydict = {'key': 'value'}
#     mytuple = (datetime.now().date().strftime('%Y-%m-%d'), 'tupla1', 'tupla2','tupla3')
#     return render_template('user.html', name=name, myindex=ind, mylist=mylist, mydict=mydict, mytuple=mytuple)


# @app.route('/segundafuncion')
# def index1():
#     print('esto se imprime solo en la terminal')
#     print(app.config['VARIABLE2']) #importar config
#     print(app.config['VARIABLE'])
#     return '<h1>Hello, Sol!</h1>'

# @app.route(f"/user/{app.config['VARIABLE']}/<name>") #ruta dinamica coge lo que ponemos en name y lo muestra en la web 
# def user(name):
#     return "<h1>Hello, {}!</h1>".format(name)


