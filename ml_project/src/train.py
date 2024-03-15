file_path = '../data/raw/tudato.csv'  

data = pd.read_csv(file_path)
data_x = data.drop('Precios', axis=1)
y = data['Precios']


columnas_log = ['GHI', 'Gas', 'PotenciaViento']

pipeline = make_pipeline(
    ColumnTransformer(
        transformers=[
            ('log', FunctionTransformer(np.log1p, validate=True), columnas_log)  # Aplica logaritmo
        ],
        remainder='passthrough'  # Mantén las columnas no especificadas
    ),
    StandardScaler()  # Estandarizar todo el DataFrame
)

# Aplica el pipeline al DataFrame
df_log_estandarizado = pipeline.fit_transform(data_x)



from sklearn.decomposition import PCA
pca = PCA(0.95)
X95 = pca.fit_transform(data_x)

modelo = RandomForestRegressor(max_depth=17, max_features=4, n_estimators=120)
modelo.fit(data_x, y)

y_pred = modelo.predict(data_x)


# guardar el modelo
import joblib

# Guardar el modelo
joblib.dump(modelo, 'new_model.pkl')