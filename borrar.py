import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, log_loss
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# Cargar los datos desde el archivo Excel
df = pd.read_excel('C:\Users\lopez\OneDrive - Universidad Privada del Norte\Desktop\CAPSTONE TRABAJO\TODA INFO\CAPSTONE 3\CAPSTONE 3')

# Revisión de calidad de los datos
# Eliminar columnas no relevantes (Nombre, Edad, Hobbie)
df = df.drop(['Nombre', 'Edad', 'Hobbie'], axis=1)

# Revisar valores nulos y duplicados
if df.isnull().sum().any():
    print("Hay valores nulos en los datos, se procederá a eliminarlos.")
    df = df.dropna()  # Eliminar filas con valores nulos

if df.duplicated().any():
    print("Existen filas duplicadas, se procederá a eliminarlas.")
    df = df.drop_duplicates()  # Eliminar filas duplicadas

# Verificar la distribución de las clases en la variable 'Dimensión'
print("Distribución de las clases en 'Dimensión':")
print(df['Dimensión'].value_counts())

# Si hay desbalance de clases, puedes usar el parámetro 'class_weight' en el modelo o aplicar técnicas de sobre/muestreo.
# En este caso, se usará 'class_weight' = 'balanced' para balancear el impacto de las clases desbalanceadas.

# Separar características (X) y la variable objetivo (y)
X = df.drop(['Dimensión', 'Preferencia 1', 'Preferencia 2', 'Preferencia 3'], axis=1)
y = df['Dimensión']

# Codificar las variables categóricas usando Label Encoding
label_encoders = {}
for column in X.columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Guardar los label encoders para usarlos en la interfaz de Streamlit
joblib.dump(label_encoders, 'label_encoders.pkl')


#####EMPEZAMOS ENTRENAMIENTO 
# Escalado de características (aunque Random Forest no lo requiere, 
# se aplica para asegurar mejores resultados en otros modelos)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Ajuste de hiperparámetros usando GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],  # Número de árboles
    'max_depth': [None, 10, 20, 30],   # Profundidad máxima de los árboles
    'min_samples_split': [2, 5, 10],   # Mínimo de muestras para dividir un nodo
    'min_samples_leaf': [1, 2, 4],     # Mínimo de muestras por hoja
    'class_weight': ['balanced', None] # Manejo de clases desbalanceadas
}

# Crear un modelo RandomForestClassifier
modelo_rf = RandomForestClassifier(random_state=42)

# Usar GridSearchCV para encontrar los mejores parámetros
grid_search = GridSearchCV(estimator=modelo_rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Imprimir los mejores parámetros encontrados
print("Mejores parámetros:", grid_search.best_params_)

# Entrenar el modelo con los mejores parámetros
modelo_rf_opt = grid_search.best_estimator_

# Evaluar el modelo optimizado
y_pred = modelo_rf_opt.predict(X_test)

# Métricas de rendimiento
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión: {accuracy * 100:.2f}%')

# Reporte completo de clasificación
print('\nReporte de Clasificación:')
print(classification_report(y_test, y_pred))

# Matriz de confusión
print('\nMatriz de Confusión:')
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=modelo_rf_opt.classes_, yticklabels=modelo_rf_opt.classes_)
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

# Precisión (Precision)
precision = precision_score(y_test, y_pred, average='weighted')
print(f'Precisión: {precision:.2f}')

# Recuperación (Recall)
recall = recall_score(y_test, y_pred, average='weighted')
print(f'Recuperación (Recall): {recall:.2f}')

# F1-Score
f1 = f1_score(y_test, y_pred, average='weighted')
print(f'F1-Score: {f1:.2f}')

# AUC-ROC (solo si tienes un modelo de clasificación binaria o multiclase con probabilidades)
y_prob = modelo_rf_opt.predict_proba(X_test)  # Obtener probabilidades de cada clase
auc_roc = roc_auc_score(y_test, y_prob, multi_class='ovr')  # 'ovr' para clasificación multiclase
print(f'AUC-ROC: {auc_roc:.2f}')

# Log-Loss (si el modelo predice probabilidades)
logloss = log_loss(y_test, y_prob)
print(f'Log-Loss: {logloss:.2f}')

# Guardar el modelo entrenado
joblib.dump(modelo_rf_opt, 'modelo_rf_holland_opt.pkl')

print("Modelo entrenado y guardado con éxito.")


knn = KNeighborsClassifier(n_neighbors=5)  # Example: 5 neighbors
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
y_prob_knn = knn.predict_proba(X_test)

# --- Decision Tree Model ---
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)
y_pred_dtree = dtree.predict(X_test)
y_prob_dtree = dtree.predict_proba(X_test)

# --- Evaluation Metrics Function ---
def evaluate_model(y_test, y_pred, y_prob, model_name):
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1-Score': f1_score(y_test, y_pred, average='weighted'),
        'AUC-ROC': roc_auc_score(y_test, y_prob, multi_class='ovr')
    }
    return metrics

# Collect metrics
rf_metrics = evaluate_model(y_test, y_pred, y_prob, 'Random Forest')
knn_metrics = evaluate_model(y_test, y_pred_knn, y_prob_knn, 'KNN')
dtree_metrics = evaluate_model(y_test, y_pred_dtree, y_prob_dtree, 'Decision Tree')

# --- Display Comparative Table ---
results_df = pd.DataFrame([rf_metrics, knn_metrics, dtree_metrics])
print("\nComparative Metrics Table:")
print(results_df)

# --- Optional: Save metrics to CSV ---
results_df.to_csv('model_comparison.csv', index=False)

# --- Visualización de ROC Curve ---
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(model, X_test, y_test, model_name):
    y_prob = model.predict_proba(X_test)
    n_classes = len(model.classes_)
    
    plt.figure(figsize=(8, 6))
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(pd.get_dummies(y_test).iloc[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f'Clase {model.classes_[i]} (AUC = {auc(fpr, tpr):.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title(f'Curva ROC - {model_name}')
    plt.legend(loc='lower right')
    plt.show()

# --- Visualización de Accuracy Curve ---
def plot_accuracy_curve(model, X_train, X_test, y_train, y_test, model_name):
    train_accuracy = []
    test_accuracy = []
    for n in range(1, 21):
        model.set_params(n_neighbors=n) if isinstance(model, KNeighborsClassifier) else None
        model.fit(X_train, y_train)
        train_accuracy.append(model.score(X_train, y_train))
        test_accuracy.append(model.score(X_test, y_test))
        
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 21), train_accuracy, label='Train Accuracy')
    plt.plot(range(1, 21), test_accuracy, label='Test Accuracy')
    plt.xlabel('Número de Vecinos' if isinstance(model, KNeighborsClassifier) else 'Iteraciones')
    plt.ylabel('Precisión')
    plt.title(f'Curva de Precisión - {model_name}')
    plt.legend()
    plt.show()

# --- Visualización de Matrices de Confusión ---
def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=modelo_rf_opt.classes_, 
    yticklabels=modelo_rf_opt.classes_)
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.show()

# --- Visualizar todas las métricas para cada modelo ---
models = {
    'Random Forest': (modelo_rf_opt, y_pred),
    'KNN': (knn, y_pred_knn),
    'Decision Tree': (dtree, y_pred_dtree)
}

for model_name, (model, y_pred) in models.items():
    print(f'\n--- {model_name} ---')
    plot_confusion_matrix(y_test, y_pred, model_name)
    plot_roc_curve(model, X_test, y_test, model_name)
    if isinstance(model, KNeighborsClassifier):  # Solo aplica para KNN
        plot_accuracy_curve(model, X_train, X_test, y_train, y_test, model_name)
