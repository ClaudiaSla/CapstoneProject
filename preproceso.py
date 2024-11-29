import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Cargar los datos desde el archivo Excel
df = pd.read_excel('Test_Holland.xlsx')

# Revisión de calidad de los datos
# Eliminar columnas no relevantes (Nombre, Edad, Hobbie)
df_clean = df.drop(['Nombre', 'Edad', 'Hobbie'], axis=1)

# Revisar valores nulos y duplicados
if df_clean.isnull().sum().any():
    print("Hay valores nulos en los datos, se procederá a eliminarlos.")
    df_clean = df_clean.dropna()  # Eliminar filas con valores nulos
    # Alternativamente, si no quieres eliminar, podrías imputar:
    # df_clean.fillna(df_clean.mean(), inplace=True)

if df_clean.duplicated().any():
    print("Existen filas duplicadas, se procederá a eliminarlas.")
    df_clean = df_clean.drop_duplicates()  # Eliminar filas duplicadas

# Verificar la distribución de las clases en la variable 'Dimensión'
print("Distribución de las clases en 'Dimensión':")
print(df_clean['Dimensión'].value_counts())

# Separar características (X) y la variable objetivo (y)
X = df_clean.drop(['Dimensión', 'Preferencia 1', 'Preferencia 2', 'Preferencia 3'], axis=1)
y = df_clean['Dimensión']

# Codificar las variables categóricas usando Label Encoding
label_encoders = {}
for column in X.columns:
    if X[column].dtype == 'object':  # Asegurarse de codificar solo las variables categóricas
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

# Escalado de características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Crear un DataFrame con las características escaladas y la variable objetivo
df_final = pd.DataFrame(X_scaled, columns=X.columns)
df_final['Dimensión'] = y

# Exportar la data preprocesada a un archivo Excel
df_final.to_excel('Data_Preprocesada.xlsx', index=False)

print("Los datos preprocesados se han guardado exitosamente en 'Data_Preprocesada.xlsx'.")
