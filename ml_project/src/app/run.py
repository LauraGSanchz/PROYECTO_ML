"""""archivo donde corre la app"""
from flask import Flask, render_template
import os
from datetime import datetime

app = Flask(__name__, instance_relative_config=True)
app.config.from_object("config")
# app.config.from_pyfile("..\config.py")
# app.config.from_envvar("APP_CONFIG_FILE")

@app.route('/')
def home():
    return render_template('home.html')

@app.route("/user/<name>/<int:ind>") #ruta dinamica coge lo que ponemos en name y lo muestra en la web 
def funcion(name, ind):
    mylist = ['elemento', 'elemento1', 'elemento2','elemento3']
    mydict = {'key': 'value'}
    mytuple = (datetime.now().date().strftime('%Y-%m-%d'), 'tupla1', 'tupla2','tupla3')
    return render_template('user.html', name=name, myindex=ind, mylist=mylist, mydict=mydict, mytuple=mytuple)


@app.route('/segundafuncion')
def index1():
    print('esto se imprime solo en la terminal')
    print(app.config['VARIABLE2']) #importar config
    print(app.config['VARIABLE'])
    return '<h1>Hello, Sol!</h1>'

@app.route(f"/user/{app.config['VARIABLE']}/<name>") #ruta dinamica coge lo que ponemos en name y lo muestra en la web 
def user(name):
    return "<h1>Hello, {}!</h1>".format(name)




if __name__ == '__main__':
    app.run(port=5000, debug=True) #debug false hace que no se actualice la app y no se vean los cambos automaticamente en la web. No se puede dejar en true cuando subo la app. 