# Tratamiento de datos
import pandas as pd
import numpy as np


# Visualización
import matplotlib.pyplot as plt
import seaborn as sns


# Modelos

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error,mean_absolute_error
from sklearn.model_selection import GridSearchCV

# skforecast
from skforecast.Sarimax import Sarimax
from skforecast.ForecasterSarimax import ForecasterSarimax
from skforecast.model_selection_sarimax import backtesting_sarimax
from skforecast.model_selection_sarimax import grid_search_sarimax
from sklearn.metrics import mean_absolute_error
from pmdarima import ARIMA
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import backtesting_forecaster
import pickle

def load_data(file_path):
    """
    Carga los datos desde un archivo CSV en un DataFrame de Pandas.
    """
    df = pd.read_csv(file_path)
    df= pd.DataFrame(df)
    df.rename(columns={0: 'values'}, inplace=True)
    return df

def preprocess_data(df):
    """
    Preprocesa los datos eliminando duplicados, reindexando y rellenando valores faltantes.
    """
    df = df[~df.index.duplicated()]
    df = df.asfreq('H')
    df.fillna(0, inplace=True)
    return df

def split_data(df, split_date='2022-12-31'):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    """
    split_index = pd.to_datetime(split_date)
    df_train = df.loc[df.index <= split_index]
    df_test = df.loc[df.index > split_index]
    return df_train, df_test

def train_model(df_train):
    """
    Entrena el modelo SARIMAX con los datos de entrenamiento.
    """
    forecaster = ForecasterSarimax(
                 regressor=Sarimax(
                                order=(1, 1, 1),
                                seasonal_order=(1, 2, 1, 12),
                                maxiter=200
                            ))

    forecaster.fit(y=df_train['values'], suppress_warnings=True)
    
    return forecaster

def calculate_mae(forecaster, df_test):
    """
    Calcula el error absoluto medio (MAE) del modelo ForecasterSarimax.

    Args:
    - forecaster: El modelo ForecasterSarimax entrenado.
    - df_test: El DataFrame de prueba con la serie temporal.

    Returns:
    - mae: El error absoluto medio (MAE) calculado.
    """
    # Calcula las predicciones del modelo
    predictions = forecaster.predict(steps=36)

    # Calcula el MAE
    mae = mean_absolute_error(df_test['values'], predictions)
    return mae

def save_model(forecaster, file_name='forecaster_model.pkl'):
    """
    Guarda el modelo entrenado en un archivo pickle.
    """
    with open(file_name, 'wb') as file:
        pickle.dump(forecaster, file)

if __name__ == "__main__":
    # Especifica la ruta relativa al archivo CSV
    file_path = 'ml_project/src/data/processed/serie_temporal.csv'  # Usando '/' para rutas en lugar de '\'

    # Cargar datos
    df = load_data(file_path)

    # Preprocesar datos
    df = preprocess_data(df)

    # Dividir datos en conjuntos de entrenamiento y prueba
    df_train, df_test = split_data(df)

    # Entrenar modelo
    forecaster = train_model(df_train)

    # Evaluar modelo
    metric = evaluate_model(forecaster, df_test)
    print(f"Métrica (error absoluto medio): {metric}")

    # Guardar modelo entrenado
    save_model(forecaster)