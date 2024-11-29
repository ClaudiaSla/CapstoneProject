import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Cargar los datos
df = pd.read_excel('Test_Holland.xlsx')
df = df.drop(['Nombre', 'Edad', 'Hobbie'], axis=1).dropna().drop_duplicates()

# Separar características (X) y la variable objetivo (y)
X = df.drop(['Dimensión', 'Preferencia 1', 'Preferencia 2', 'Preferencia 3'], axis=1)
y = df['Dimensión']

# Codificación de variables categóricas
label_encoders = {}
for column in X.columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le
joblib.dump(label_encoders, 'label_encoders.pkl')

# Escalar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modelos
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=20, random_state=42)
}

# Inicializar diccionario para guardar resultados
results = []

# Evaluación y graficación
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # --- Accuracy Curve ---
    train_sizes = np.linspace(0.1, 0.9, 10)  # Cambié el rango a 0.9 en lugar de 1.0
    train_accuracies = []
    
    for size in train_sizes:
        X_partial, _, y_partial, _ = train_test_split(X_train, y_train, train_size=size, random_state=42)
        model.fit(X_partial, y_partial)
        y_partial_pred = model.predict(X_test)
        train_accuracies.append(accuracy_score(y_test, y_partial_pred))
    
    plt.plot(train_sizes * 100, train_accuracies, label=name)
    plt.title(f'Accuracy Curve - {name}')
    plt.xlabel('Training Set Size (%)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

    # --- Matriz de Confusión ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión - {name}')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.show()

    # Guardar las métricas para el cuadro comparativo
    accuracy = accuracy_score(y_test, y_pred)
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Confusion Matrix": cm
    })

# Crear cuadro comparativo
results_df = pd.DataFrame(results)

# Mostrar resultados en formato de tabla
print("Cuadro Comparativo de Modelos:")
print(results_df[['Model', 'Accuracy']])

# También puedes guardar los resultados en un archivo CSV si es necesario
results_df.to_csv('model_comparison_results.csv', index=False)
