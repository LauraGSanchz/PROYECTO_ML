"""""archivo donde corre la app"""
from flask import Flask, request, render_template,jsonify
import os
import seaborn as sns
import pandas as pd
from joblib import  load
import statsmodels.api as sm

app = Flask(__name__, static_url_path='/static')


modelo = None
modelo_ruta = 'C:\\Users\\laura\\OneDrive\\Desktop\\PROYECTO ML\\ml_project\\src\\models\\ignore\\exogen_model3.joblib'

def cargar_modelo():
    global modelo
    if os.path.exists(modelo_ruta):
        with open(modelo_ruta, 'rb') as archivo_modelo:
            modelo = load(archivo_modelo)
    else:
        print("El archivo del modelo no existe en la ruta especificada.")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_form')
def prediction_form():
    return render_template('predict.html')

@app.route('/exploracion', methods=['GET'])
def exploracion():
    return render_template('explo.html')

@app.route('/exploracion2', methods=['GET'])
def exploracion2():
    return render_template('explo2.html')



@app.route('/predict', methods=['POST'])
def predict():
    global modelo
    
    # Obtener el número de pasos del formulario
    steps = request.form.get('steps')
    
    # Obtener el archivo CSV exógeno del formulario
    file = request.files['data']
    
    # Verificar si todos los campos requeridos están presentes
    if None in (steps, file):
        return jsonify({'error': 'Faltan campos requeridos en el formulario.'}), 400
    
    # Convertir los pasos a un entero
    steps = int(steps)
    
    # Leer el archivo CSV como DataFrame
    try:
        df_data = pd.read_csv(file)
        df_data = pd.DataFrame(df_data)
    except Exception as e:
        return jsonify({'error': 'Error al leer el archivo CSV: ' + str(e)}), 400
    
    # Cargar el modelo si no se ha cargado aún
    if modelo is None:
        cargar_modelo()
    
    if modelo is not None:
        # Ahora, utiliza el número de pasos recibido y los datos exógenos para realizar la predicción
        predictions = modelo.predict(steps=steps, exog=df_data[['GHI', 'Festivo', 'Gas', 'PotenciaViento']])
        # Devolver los resultados como una respuesta JSON
        return jsonify({'predictions': list(predictions)})
    else:
        return jsonify({'error': 'El modelo no pudo ser cargado.'}), 500



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


