import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib.pyplot as plt

def leerLimpiar():
    url = 'Carpetas_Set 20240328.xlsx'
    df = pd.read_excel(url, sheet_name='Energia', skiprows=1, usecols="C:D,H:I")
    df_limpio = df[(df['CoordX'] != 0) & (df['CoordY'] != 0)].drop_duplicates(subset='Nombre').copy()
    return df_limpio

def predict(df, num_grupos):
    kmeans = KMeans(n_clusters=num_grupos, random_state=42)
    df['Grupo'] = kmeans.fit_predict(df[['CoordX', 'CoordY']])
    
    # Asignar letras a los grupos en lugar de números
    df['Grupo'] = df['Grupo'].apply(lambda x: chr(x + 65))  # Asignar letras A, B, C, etc.
    
    # Rebalanceo de los grupos
    max_empresas_por_grupo = len(df) // num_grupos
    df_equilibrado = pd.DataFrame(columns=df.columns)

    for grupo in range(num_grupos):
        empresas_grupo = df[df['Grupo'] == chr(grupo + 65)]
        if len(empresas_grupo) > max_empresas_por_grupo:
            empresas_grupo = empresas_grupo.sample(max_empresas_por_grupo, random_state=42)
        
        df_equilibrado = pd.concat([df_equilibrado, empresas_grupo])
    
    df_equilibrado['Orden'] = df_equilibrado.groupby('Grupo').cumcount() + 1
    df_equilibrado['NombreSectorizado'] = df_equilibrado['Grupo'].astype(str) + df_equilibrado['Orden'].astype(str)
    
    return df_equilibrado

def optimizador(df):
    rutas_optimas = {}
    estimado = {}
    velocidad = 5  # Velocidad en km/h
    tiempoEstimado = 4  # 4 horas para 150 paradas

    for grupo in df['Grupo'].unique():
        subgrupo = df[df['Grupo'] == grupo]
        coordenadas = subgrupo[['CoordX', 'CoordY']].drop_duplicates()

        if len(coordenadas) < 2:
            print(f"Grupo {grupo} tiene menos de 2 coordenadas únicas, se omite la optimización.")
            continue

        G = nx.complete_graph(len(coordenadas))
        distancias = {(i, j): np.linalg.norm(np.array(coordenadas.iloc[i]) - np.array(coordenadas.iloc[j])) for i in range(len(coordenadas)) for j in range(len(coordenadas))}
        nx.set_edge_attributes(G, distancias, 'weight')

        tsp_ruta = nx.approximation.traveling_salesman_problem(G, weight='weight')
        rutas_optimas[grupo] = tsp_ruta

        # Calcular la distancia total de la ruta
        distancia_total_km = sum(G.edges[tsp_ruta[i], tsp_ruta[i + 1]]['weight'] for i in range(len(tsp_ruta) - 1))

        # Calcular el tiempo de caminata basado en la distancia total
        tiempo_caminata_horas = distancia_total_km / velocidad

        numero_paradas = len(coordenadas)
        tiempo_estimado_horas = (numero_paradas / 150) * tiempoEstimado + tiempo_caminata_horas

        horas = int(tiempo_estimado_horas)
        minutos = int((tiempo_estimado_horas - horas) * 60)

        estimado[grupo] = f"{horas}h {minutos}m"

    return rutas_optimas, estimado

def visualizarRutas(df, rutas):
    plt.figure(figsize=(12, 8))
    
    for grupo, ruta in rutas.items():
        subgrupo = df[df['Grupo'] == grupo].reset_index(drop=True)
        coord_ruta = subgrupo.iloc[ruta][['CoordX', 'CoordY']]

        plt.plot(coord_ruta['CoordX'], coord_ruta['CoordY'], marker='o', label=f'Ruta {grupo}')

    plt.title('Rutas de Control de Medidores')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.legend()
    plt.grid()
    plt.show()

def empresas(df):
    conteo = df['Grupo'].value_counts()

    plt.figure(figsize=(10, 6))
    conteo.plot(kind='bar', color='skyblue')
    plt.title('Cantidad de Empresas por Grupo')
    plt.xlabel('Grupo')
    plt.ylabel('Cantidad de Empresas')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.show()

df = leerLimpiar()
df = df.head(1000)
df = predict(df, num_grupos=12)
rutas, tiempo = optimizador(df)
empresas(df)
visualizarRutas(df, rutas)

for grupo, tiempo_grupo in tiempo.items():
    print(f"Grupo {grupo}: estimada de = {tiempo_grupo} horas")
