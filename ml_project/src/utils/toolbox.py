def explore_data(df):
    print('Data Set Shape:',df.shape)
    print('-------'*100)
    print('Data Columns list:',df.columns)
    print('-------'*100)
    print(df.isna().sum())
    print('-------'*100)
    df.info()


def duplicated_info(df):
    print(df.duplicated().sum())
    print(df.duplicated().value_counts())
    df[df.duplicated()].sort_values(by='column')
    df.loc[df.duplicated()]

def explore_data(data):
    """
    Función para explorar los datos cargados.
    """
    # Muestra las primeras filas del DataFrame
    print("Primeras filas del DataFrame:")
    print(data.head())

    # Información sobre las columnas y tipos de datos
    print("\nInformación del DataFrame:")
    print(data.info())

    # Resumen estadístico de las variables numéricas
    print("\nResumen estadístico de variables numéricas:")
    print(data.describe())

    # Resumen de las variables categóricas
    print("\nResumen de variables categóricas:")
    print(data.describe(include='object'))




def plot_distribution(data, column):
    """
    Función para trazar la distribución de una columna.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True)
    plt.title(f'Distribución de {column}')
    plt.xlabel(column)
    plt.ylabel('Frecuencia')
    plt.show()

def plot_correlation_heatmap(data):
    """
    Función para trazar un mapa de calor de correlación.
    """
    # Create a mask to hide the upper triangle (including the diagonal)
    mask = np.tri(corr_matrix.shape[0], k=-1, dtype=bool)
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, mask=mask,cmap='coolwarm', fmt=".2f")
    plt.title('Mapa de calor de correlación')
    plt.show()


def plot_boxplots(df, columns_per_row):
    #VARIOS GRAFICOS EN LA MISMA FIGURA##

      # Iterate over each column and create a separate boxplot
    num_columns = len(df.columns)
    num_rows = -(-num_columns // columns_per_row)  # Ceiling division to calculate the number of rows needed
    fig, axes = plt.subplots(num_rows, columns_per_row, figsize=(12, 6))

    # Flatten the axes array if only one row is needed
    if num_rows == 1:
        axes = axes.reshape(1, -1)

    for i, column in enumerate(df.columns):
        row = i // columns_per_row
        col = i % columns_per_row
        sns.boxplot(data=df[column], ax=axes[row, col])
        axes[row, col].set_title(f'Boxplot of {column}')

    # Hide any unused subplots
    for i in range(num_columns, num_rows * columns_per_row):
        row = i // columns_per_row
        col = i % columns_per_row
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.show()

# for column in data.columns:
#     plt.figure(figsize=(8, 6))
#     sns.boxplot(data=df[column])
#     plt.title(f'Boxplot of {column}')
#     plt.show()


def histogram_group(columns_per_row: int, columns: list): # Nº de graficos por fila
    "La función crea histgramas con las columnas que le indiquemos"
   
    'Ancho de cada grfico'
    plot_width = 8

    'Alto de cada grafico basado en un ratio de aspecto(se ajusta para controlar el alto)'
    aspect_ratio = 0.75  
    plot_height = plot_width * aspect_ratio

    'Iterar sobre cada columna y crear un histograma'
    num_columns = len(columns)
    num_rows = -(-num_columns // columns_per_row)  "División de techo para calcular el número de filas necesarias"
    fig, axes = plt.subplots(num_rows, columns_per_row, figsize=(plot_width * columns_per_row, plot_height * num_rows))

     'Aplanar la matriz de ejes si solo se necesita una fila'
    if num_rows == 1:
        axes = axes.reshape(1, -1)

    for i, column in enumerate(columns):
        row = i // columns_per_row
        col = i % columns_per_row
        sns.histplot(data=df, x=column, kde=True, ax=axes[row, col])
        axes[row, col].set_title(f'Histogram of {column}')

    '# Hide any unused subplots'
    for i in range(num_columns, num_rows * columns_per_row):
        row = i // columns_per_row
        col = i % columns_per_row
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    return plt.show()




# EDA

from pandas_profiling import ProfileReport
profile = ProfileReport(df, title="Report")
profile
------------------------------------------------------------------------------------------------------------
# Para cuando tengamos fallos con las rutas
import sys  (interactua con el sistema operativo)
sys.path.append("./src/notebooks/")

import sys 
sys.path.append("..")


from carpeta.notebook import función

from pathlib import Path (interaactua con las rutas de python)

sys.path.append(ruta) para traer funciones de otras carpetas.
 from carpeta import funcion


------------------------------------------------------------------------------------------------------------

# PERIODOGRAMA
def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = pd.Timedelta("365D") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax


# SEASONAL PLOT(series temporales)
def seasonal_plot(X, y, period, freq, ax=None):
    # Verificar que period y freq sean columnas en X
    assert period in X.columns, f"'{period}' no está en las columnas de X"
    assert freq in X.columns, f"'{freq}' no está en las columnas de X"
    assert y in X.columns, f"'{y}' no está en las columnas de X"

    if ax is None:
        _, ax = plt.subplots()

    palette = sns.color_palette("husl", n_colors=X[period].nunique())

    # Convertir los datos a arrays de NumPy
    x_data = X[freq].values
    y_data = X[y].values
    hue_data = X[period].values

    ax = sns.lineplot(
        x=x_data,
        y=y_data,
        hue=hue_data,
        ci=False,
        ax=ax,
        palette=palette,
        legend=False,
    )
    ax.set_title(f"Seasonal Plot ({period}/{freq})")

    # Ajustar las anotaciones
    unique_periods = np.unique(hue_data)
    for line, name in zip(ax.lines, unique_periods):
        y_ = line.get_ydata()[-1]
        ax.annotate(
            name,
            xy=(1, y_),
            xytext=(6, 0),
            color=line.get_color(),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            size=14,
            va="center",
        )

    return ax
# ONE HOT ENCODER

def apply_onehot_encoder(train:pd.DataFrame, columns_to_encode:list, test:pd.DataFrame=None):
    
    # Resetear índices para evitar desalineación
    train = train.reset_index(drop=True)
    
    # Crear el OneHotEncoder
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    # Ajustar y transformar las columnas seleccionadas
    transformed_data = encoder.fit_transform(train[columns_to_encode])

    # Crear un DataFrame con las columnas transformadas
    transformed_df = pd.DataFrame(transformed_data, columns=encoder.get_feature_names_out(columns_to_encode))
    
    # Concatenar con el DataFrame original excluyendo las columnas transformadas
    df_concatenated = pd.concat([train.drop(columns_to_encode, axis=1), transformed_df], axis=1)

    # Si se proporciona un segundo DataFrame, aplicar la misma transformación
    if test is not None:
        transformed_data_to_transform = encoder.transform(test[columns_to_encode])
        transformed_df_to_transform = pd.DataFrame(transformed_data_to_transform, columns=encoder.get_feature_names_out(columns_to_encode))
        df_to_transform_concatenated = pd.concat([test.drop(columns_to_encode, axis=1), transformed_df_to_transform], axis=1)
        return df_concatenated, df_to_transform_concatenated

    return df_concatenated