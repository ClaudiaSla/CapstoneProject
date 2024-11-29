import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cargar el modelo y los label encoders
modelo_rf = joblib.load('modelo_rf_holland_opt.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Descripciones y características principales de cada dimensión
descripcion_tipos = {
    "Realista": "Prefiere actividades manuales, mecánicas y al aire libre. Poco sociable y materialista.",
    "Investigador": "Disfruta investigar y resolver problemas complejos. Analítico y crítico.",
    "Artístico": "Creativo y emocional, prefiere actividades artísticas y desordenadas.",
    "Social": "Orienta sus habilidades al servicio de los demás, idealista y sociable.",
    "Convencional": "Prefiere tareas organizadas y estructuradas. Eficiente y práctica.",
    "Emprendedor": "Líder natural, busca desafíos y logros económicos. Ambicioso y confiado."
}

carreras_recomendadas = {
    "Realista": ["Construcción", "Agronomía", "Ingeniería Mecánica"],
    "Investigador": ["Física", "Matemáticas", "Química"],
    "Artístico": ["Arte", "Música", "Diseño Gráfico"],
    "Social": ["Psicología", "Pedagogía", "Trabajo Social"],
    "Convencional": ["Administración de Empresas", "Contabilidad", "Finanzas"],
    "Emprendedor": ["Administración de Negocios", "Marketing", "Derecho Empresarial"]
}

caracteristicas_tipos = {
    "Realista": {"Poco sociable": 8, "Materialista": 7, "Persistente": 9, "Estable": 8, "Sincero": 7, "Ahorrativo": 6},
    "Investigador": {"Analítico": 9, "Curioso": 8, "Racional": 9, "Crítico": 7, "Intelectual": 9, "Pesimista": 5},
    "Artístico": {"Imaginativa": 9, "Original": 9, "Intuitiva": 8, "Emocional": 7, "Independiente": 8, "Idealista": 7},
    "Social": {"Influyente": 8, "Servicial": 9, "Idealista": 7, "Sociable": 9, "Amable": 8, "Persuasiva": 7},
    "Convencional": {"Conformista": 8, "Inhibida": 6, "Ordenada": 9, "Eficiente": 8, "Práctica": 7, "Persistente": 8},
    "Emprendedor": {"Dominante": 9, "Optimista": 8, "Enérgica": 9, "Sociable": 8, "Confiada en sí misma": 9, "Ambiciosa": 8}
}

# Preguntas del test
preguntas = [
    "¿Te interesa reparar cosas por tu cuenta?",
    "¿Disfrutas resolver problemas matemáticos o científicos?",
    "¿Te gusta dibujar, pintar o hacer manualidades?",
    "¿Prefieres ayudar a otros con sus problemas personales?",
    "¿Te ves liderando un grupo o proyecto?",
    "¿Te gusta mantener todo organizado y en su lugar?",
    "¿Te interesa trabajar al aire libre, como en jardines o construcciones?",
    "¿Disfrutas investigar para aprender más sobre algo?",
    "¿Te gusta actuar, cantar o bailar en público?",
    "¿Te gustaría ser mentor o enseñar a otros?",
    "¿Te interesa iniciar tu propio negocio?",
    "¿Prefieres seguir reglas claras y definidas?",
    "¿Te gusta trabajar con herramientas o maquinaria?",
    "¿Disfrutas diseñar experimentos o probar teorías?",
    "¿Te interesa escribir historias o poesía?",
    "¿Disfrutas colaborar con otras personas en proyectos grupales?",
    "¿Te sientes cómodo tomando decisiones importantes?",
    "¿Prefieres tareas con procedimientos claros y repetitivos?",
    "¿Te interesa construir cosas desde cero?",
    "¿Disfrutas leer libros de ciencia o tecnología?",
    "¿Te gusta asistir a museos o exposiciones de arte?",
    "¿Prefieres hablar con personas para resolver conflictos?",
    "¿Te gusta convencer a otros de tus ideas?",
    "¿Disfrutas trabajar con números y estadísticas?",
    "¿Te interesa la agricultura o el trabajo en el campo?",
    "¿Disfrutas buscar respuestas a preguntas complejas?",
    "¿Te gusta crear diseños gráficos o visuales?",
    "¿Te sientes feliz ayudando en actividades comunitarias?",
    "¿Prefieres organizar eventos o actividades grupales?",
    "¿Te sientes cómodo con tareas administrativas, como archivar documentos?"
]

# Título de la aplicación
st.title("Test Vocacional Holland con IA")
st.write("Responde las siguientes preguntas para conocer tu dimensión vocacional y carreras recomendadas:")

# Crear un formulario con las preguntas
respuestas = []
opciones = ["Sí", "No", "Tal vez"]

with st.form("formulario_test"):
    for pregunta in preguntas:
        respuesta = st.selectbox(pregunta, opciones, key=pregunta)
        respuestas.append(respuesta)
    submit_button = st.form_submit_button("Predecir")

# Procesar la entrada del usuario y hacer la predicción
if submit_button:
    # Convertir las respuestas a formato numérico usando los label encoders
    respuestas_encoded = []
    for i, respuesta in enumerate(respuestas):
        columna = preguntas[i]
        respuestas_encoded.append(label_encoders[columna].transform([respuesta])[0])
    
    # Crear un DataFrame con las respuestas codificadas
    df_respuestas = pd.DataFrame([respuestas_encoded], columns=label_encoders.keys())
    
    # Hacer la predicción
    prediccion_proba = modelo_rf.predict_proba(df_respuestas)[0]
    prediccion_dimension = modelo_rf.classes_[prediccion_proba.argmax()]
    
    # Mostrar el resultado de la predicción
    st.subheader("Resultados de la Predicción:")
    st.write(f"**Dimensión Predicha:** {prediccion_dimension}")
    
    # Mostrar las probabilidades en forma de gráfica
    dimensiones = modelo_rf.classes_
    plt.figure(figsize=(10, 6))
    sns.barplot(x=dimensiones, y=prediccion_proba, palette="viridis")
    plt.title("Comparativa de Dimensiones")
    plt.ylabel("Probabilidad")
    plt.xlabel("Dimensiones")
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # Mostrar descripción y carreras recomendadas
    st.write("### Descripción del Tipo Predicho:")
    st.write(descripcion_tipos[prediccion_dimension])

    st.write("### Carreras Recomendadas:")
    for carrera in carreras_recomendadas[prediccion_dimension]:
        st.write(f"- {carrera}")

    # Gráfico de radar para la dimensión predicha
    def crear_radar(data, titulo):
        etiquetas = list(data.keys())
        valores = list(data.values())
        valores.append(valores[0])  # Cerrar el gráfico
        
        angulos = np.linspace(0, 2 * np.pi, len(etiquetas), endpoint=False).tolist()
        angulos += angulos[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(angulos, valores, color='blue', alpha=0.25)
        ax.plot(angulos, valores, color='blue', linewidth=2)
        ax.set_xticks(angulos[:-1])
        ax.set_xticklabels(etiquetas)
        ax.set_title(titulo, size=15, color='navy', pad=20)
        st.pyplot(fig)

    crear_radar(caracteristicas_tipos[prediccion_dimension], f"Características del Tipo {prediccion_dimension}")
