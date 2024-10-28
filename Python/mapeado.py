import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib.pyplot as plt
import osmnx as ox
import matplotlib.cm as cm

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
    estimado = {}  # Dicciónario para almacenar el tiempo estimado de cada grupo
    
    velocidad = 5  # Velocidad en km/h
    tiempo_estimado_por_150_paradas = 4  # 4 horas para 150 paradas

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

        # Ajustar el tiempo total usando el valor estimado de 4 horas para 150 paradas
        numero_paradas = len(coordenadas)
        tiempo_estimado_horas = (numero_paradas / 150) * tiempo_estimado_por_150_paradas + tiempo_caminata_horas

        horas = int(tiempo_estimado_horas)
        minutos = int((tiempo_estimado_horas - horas) * 60)

        estimado[grupo] = f"{horas}h {minutos}m"

    return rutas_optimas, estimado

def visualizarRutas(df, rutas):
    centro = (-30.9918586, -64.0970209)  # Coordenadas de la cooperativa
    radio = 10000  # Aumentar el radio a 10,000 metros
    G = ox.graph_from_point(centro, dist=radio, network_type='all')

    # Crear la columna 'Nodo' en el DataFrame
    df['Nodo'] = df.apply(lambda row: ox.distance.nearest_nodes(G, row['CoordY'], row['CoordX']), axis=1)

    fig, ax = ox.plot_graph(G, show=False, close=False)
    ax.scatter(centro[1], centro[0], color='red', s=100, label='Cooperativa', zorder=5)

    colores = cm.rainbow(np.linspace(0, 1, len(rutas)))  # Generar colores únicos para cada grupo

    for color, (grupo, ruta) in zip(colores, rutas.items()):
        # Agrupar las paradas con las mismas coordenadas
        coord_paradas = df[df['Grupo'] == grupo][['CoordX', 'CoordY']].drop_duplicates()

        # Marcar los puntos de parada (nodos) en el mapa
        for idx, row in coord_paradas.iterrows():
            ax.scatter(row['CoordX'], row['CoordY'], color=color, s=100, zorder=6, label=f'Grupo {grupo}' if idx == 0 else "")  # Usar el color del grupo

        # Dibujar las rutas entre los nodos de parada
        for i in range(len(ruta) - 1):
            nodo_actual = df.loc[df['Grupo'] == grupo, 'Nodo'].values[ruta[i]]
            nodo_siguiente = df.loc[df['Grupo'] == grupo, 'Nodo'].values[ruta[i + 1]]

            # Verificar si ambos nodos están en el grafo y calcular la ruta más corta
            if nodo_actual in G and nodo_siguiente in G:
                ruta_mas_corta = nx.shortest_path(G, nodo_actual, nodo_siguiente, weight='length')
                x_coords = [G.nodes[n]['x'] for n in ruta_mas_corta]
                y_coords = [G.nodes[n]['y'] for n in ruta_mas_corta]

                # Dibujar la ruta más corta entre nodos en el grafo
                ax.plot(x_coords, y_coords, color=color, alpha=0.7)

    # Agregar leyenda solo de grupos y colores
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=grupo, markersize=10, markerfacecolor=color) for grupo, color in zip(rutas.keys(), colores)]
    ax.legend(handles=handles, title="Grupos")
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

# Cargar, predecir y optimizar
df = leerLimpiar()
df = df.head(1000)  # Usar solo los primeros 1000 datos
df = predict(df, num_grupos=12)
rutas, tiempo = optimizador(df)

# Visualizar resultados
empresas(df)
visualizarRutas(df, rutas)

# Mostrar tiempo estimado por grupo
for grupo, tiempo_grupo in tiempo.items():
    print(f"Grupo {grupo}: estimada de = {tiempo_grupo} horas")