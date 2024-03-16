"""""archivo donde corre la app"""
from flask import Flask

app = Flask(__name__, instance_relative_config=True)
app.config.from_object("config")
# app.config.from_pyfile("..\config.py")

@app.route('/')
def index():
    print(app.config['VARIABLE2']) #importar config
    # print(app.config['BCRYPT_LOG_ROUNDS'])
    return '<h1>Hello, Sol!</h1>'


@app.route('/segundafuncion')
def index1():
    print('esto se imprime solo en la terminal')
    return '<h1>segunda cosa!</h1>'


@app.route("/user/<name>") #ruta dinamica coge lo que ponemos en name y lo muestra en la web 
def user(name):
    return "<h1>Hello, {}!</h1>".format(name)

if __name__ == '__main__':
    app.run(port=5000, debug=True) #debug false hace que no se actualice la app y no se vean los cambos automaticamente en la web. No se puede dejar en true cuando subo la app. 