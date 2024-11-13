import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def leerLimpiar():
    url = r'Ciencia de Datos\Dataset_Estudiantes.csv'
    columnas = ['Edad','Genero','Puntaje matematicas','Puntaje Lengua','Asistencia']
    df = pd.read_csv(url, sep=';', usecols=columnas)
    df = df.dropna()
    
    return df

def horasEstudio(df):
    # Rango de horas de estudio para los puntajes más altos
    max = 16
    min = 2

    # Normalizar los puntajes de matemáticas entre 0 y 1
    df['Puntaje Normalizado'] = (df['Puntaje matematicas'] - df['Puntaje matematicas'].min()) / (df['Puntaje matematicas'].max() - df['Puntaje matematicas'].min())
    
    # Generar las horas de estudio basadas en el puntaje normalizado
    df['Horas de estudio semanal'] = df['Puntaje Normalizado'] * (max - min) + min

    # Añadir un factor de aleatoriedad para simular variabilidad
    df['Horas de estudio semanal'] += np.random.normal(0, 1.5, len(df))

    # Redondear a una hora cercana
    df['Horas de estudio semanal'] = df['Horas de estudio semanal'].clip(lower=min, upper=max).round()

    # Eliminar columna de puntaje normalizado si no es necesaria
    df.drop(columns=['Puntaje Normalizado'], inplace=True)
    
    df.to_csv('Ciencia de Datos/Dataset_Estudiantes.csv', sep=';', index=False)

    return df

def graficos(df):
    #Comparando según genero
    promedio = df.groupby('Genero')[['Puntaje matematicas', 'Puntaje Lengua']].mean()
    posiciones = np.arange(len(promedio))
    ancho = 0.4
    plt.figure(figsize=(10, 6))
    plt.bar(posiciones - ancho/2, promedio['Puntaje matematicas'], ancho, label='Matemáticas', color='skyblue')
    plt.bar(posiciones + ancho/2, promedio['Puntaje Lengua'], ancho, label='Lengua', color='lightcoral')
    plt.xlabel('Género')
    plt.ylabel('Puntaje Promedio')
    plt.title('Puntaje Promedio en Matemáticas y Lengua por Género')
    plt.xticks(posiciones, promedio.index)
    plt.legend()
    plt.show()
    

    # Gráfico de dispersión
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Puntaje matematicas'], df['Puntaje Lengua'], alpha=0.6, color='purple')
    plt.xlim(df['Puntaje matematicas'].min() - 3, df['Puntaje matematicas'].max() + 3)
    plt.ylim(df['Puntaje Lengua'].min() - 3, df['Puntaje Lengua'].max() + 3)
    plt.plot([0, 103], [0, 103], color='red', linestyle='--', linewidth=2, label="Igual Puntaje")
    plt.title("Relación entre Matemáticas y Lengua")
    plt.xlabel("Puntaje en Matemáticas")
    plt.ylabel("Puntaje en Lengua")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Grafico de Estudio
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Puntaje matematicas'], df['Horas de estudio semanal'], alpha=0.6, color='blue')
    plt.title("Relación entre Calificación en Matemáticas y Horas de Estudio")
    plt.xlabel("Puntaje en Matemáticas")
    plt.ylabel("Horas de Estudio Semanal")
    plt.grid(True)
    plt.show()

def asimetria(df):
    if df[['Puntaje matematicas', 'Asistencia', 'Edad', 'Horas de estudio semanal']].isnull().any().any():
        print("Existen valores nulos en las variables. Asegúrate de tratarlos antes de calcular la asimetría.")
        return
    
    # Asimetría de Puntaje matemáticas por Género
    matematicas = df.groupby('Genero')['Puntaje matematicas'].skew()
    asistencia = df['Asistencia'].skew()
    edad = df['Edad'].skew()
    estudio = df['Horas de estudio semanal'].skew()

    print("Asimetría del Puntaje en Matemáticas por Género:")
    for genero, asimetria_valor in matematicas.items():
        print(f"{genero}: {asimetria_valor:.2f}")
    
    print(f"\nAsimetría de la Asistencia: {asistencia:.2f}")
    print(f"Asimetría de la Edad: {edad:.2f}")
    print(f"Asimetría de las Horas de estudio semanal: {estudio:.2f}")

def regresionMultiple(df):
    X = df[['Edad', 'Asistencia', 'Horas de estudio semanal']]
    Y = df['Puntaje matematicas']

    # Escalar las variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dividir en entrenamiento y prueba (80% - 20%)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

    # Crear el modelo de regresión lineal múltiple
    modelo = LinearRegression()

    # Entrenar el modelo
    modelo.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = modelo.predict(X_test)

    # Evaluar el modelo
    print("Coeficientes:", modelo.coef_)
    print("Intersección (intercept):", modelo.intercept_)
    print("\nEvaluación del modelo:")
    print(f"R^2 (Coeficiente de determinación): {modelo.score(X_test, y_test):.2f}")
    print(f"Error Absoluto Medio (MAE): {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"Error Cuadrático Medio (MSE): {mean_squared_error(y_test, y_pred):.2f}")
    print(f"Raíz del Error Cuadrático Medio (RMSE): {mean_squared_error(y_test, y_pred, squared=False):.2f}")

def correlacion(df):
    
    df['Genero'] = df['Genero'].map({'Masculino': 1, 'Femenino': 0})
    correlaciones = df.corr()
    correlacion_puntaje_matematicas = correlaciones['Puntaje matematicas'].sort_values(ascending=False)
    print("Variables con mejor correlación con el Puntaje en Matemáticas:")
    print(correlacion_puntaje_matematicas)

def sinergias(df):
    # Excluir 'Genero' de la lista de columnas a imputar
    columnas_imputar = ['Horas de estudio semanal', 'Edad', 'Asistencia']

    # Imputación de valores faltantes
    imputer = SimpleImputer(strategy='mean')
    df[columnas_imputar] = imputer.fit_transform(df[columnas_imputar])

    # Eliminar 'Genero' de las variables predictoras
    X = df[['Horas de estudio semanal', 'Edad', 'Asistencia']]
    y = df['Puntaje matematicas']

    # Dividir en datos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Mostrar los resultados del modelo
    print("R^2 en entrenamiento:", modelo.score(X_train, y_train))
    print("R^2 en prueba:", modelo.score(X_test, y_test))

def regrecionLogistica(df):
    # Crear columna binaria para aprobación: 1 si aprueba 0 si no
    df['Aprobado'] = df['Puntaje matematicas'].apply(lambda x: 1 if x >= 70 else 0)

    # Variables predictoras y variable objetivo
    X = df[['Edad', 'Asistencia', 'Horas de estudio semanal']]
    y = df['Aprobado']

    # Escalar las variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # División de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Crear el modelo de regresión logística
    modelo = LogisticRegression()

    # Entrenar el modelo
    modelo.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = modelo.predict(X_test)

    # Evaluar el modelo
    print("Exactitud (Accuracy):", accuracy_score(y_test, y_pred))
    print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
    print("Reporte de Clasificación:\n", classification_report(y_test, y_pred))


df = leerLimpiar()
df = horasEstudio(df)
#graficos(df)
#asimetria(df)
#correlacion(df) #Compruebo que variables son las mas optimas para el analisis de datos
#sinergias(df)
#regresionMultiple(df)
regrecionLogistica(df)


#print("\n Registros completos")
#print(df)

# print("\n Descripcion")
# print(df.describe().round(2))
