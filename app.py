import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Intentar leer el archivo con diferentes codificaciones
try:
    data = pd.read_csv("dataset/data2.csv", nrows=10000, encoding='utf-8')
except UnicodeDecodeError:
    try:
        data = pd.read_csv("dataset/data2.csv", nrows=10000, encoding='latin1')
    except UnicodeDecodeError:
        data = pd.read_csv("dataset/data2.csv", nrows=10000, encoding='iso-8859-1')

# Definir la función para calcular similitud de coseno y de precios
def calcular_similitud_parcial(nombre_del_producto, data, alpha=0.7):
    # Vectorizar los nombres de los productos para coincidencias parciales
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))  # Usar n-gramas de 1 a 3 palabras
    tfidf_matrix = vectorizer.fit_transform(data['nombre'])
    
    # Calcular la matriz de similitud de coseno basada en nombres
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Calcular la similitud de precios
    precios = data['precio_actual'].fillna(data['precio_descuento']).values.reshape(-1, 1)
    max_precio = np.max(precios)  # Normalizar los precios
    precios_normalizados = precios / max_precio
    price_similarity_matrix = 1 - np.abs(precios_normalizados - precios_normalizados.T)
    
    # Combinar la similitud de coseno y de precios
    combined_similarity = alpha * similarity_matrix + (1 - alpha) * price_similarity_matrix

    # Obtener los índices de los productos que contienen la palabra clave
    producto_indices = data[data['nombre'].str.contains(nombre_del_producto, case=False, na=False)].index
    
    # Verificar si hay algún producto encontrado
    if len(producto_indices) == 0:
        st.write(f"No se encontraron productos que coincidan con '{nombre_del_producto}'.")
        return pd.DataFrame()

    # Obtener el índice del primer producto encontrado
    product_index = producto_indices[0]

    # Obtener las similitudes con otros productos
    product_similarities = combined_similarity[product_index]

    # Obtener los índices de los productos más similares
    most_similar_products_indices = np.argsort(-product_similarities)

    # Mantener un registro de los productos vistos por nombre
    seen_products = set()
    unique_indices = []

    # Seleccionar los productos más similares, ignorando duplicados de nombre
    for index in most_similar_products_indices:
        if len(unique_indices) >= 20:
            break
        product_name = data.loc[index, 'nombre']
        if product_name not in seen_products:
            seen_products.add(product_name)
            unique_indices.append(index)

    most_similar_products = data.loc[unique_indices, ['nombre', 'precio_actual', 'precio_descuento', 'supermarket', 'imagen_url', 'link']]

    return most_similar_products

# Configurar la aplicación con título y descripción
st.title('Consulta de Recomendación de Productos')
st.write('Esta aplicación permite realizar consultas de recomendación de productos utilizando un modelo de similitud de coseno basado en nombres y precios.')

# Interfaz para ingresar el nombre del producto con un placeholder personalizado
st.text_input('Escriba el nombre del producto:', value="", key="producto_input", placeholder="Introduzca el nombre del producto aquí", on_change=None)

# Crear un botón para realizar la búsqueda
buscar = st.button('Buscar')

# Crear un contenedor dinámico para los resultados
resultado_contenedor = st.empty()

# Evitar ejecutar búsqueda en cada cambio de texto
if buscar:  # Solo se ejecuta cuando se presiona el botón
    nombre_del_producto = st.session_state["producto_input"]
    
    if nombre_del_producto:
        # Limpiar el contenido del contenedor antes de mostrar los resultados
        resultado_contenedor.empty()  # Asegurarse de limpiar antes de mostrar nuevos resultados

        # Normalizar la entrada del usuario
        palabras = nombre_del_producto.lower().split()

        # Crear una expresión regular para buscar productos que contengan todas las palabras
        regex = '.*' + '.*'.join(palabras) + '.*'

        # Filtrar la lista de productos basada en la entrada del usuario (LIKE %nombre%)
        productos_filtrados = data[data['nombre'].str.contains(regex, case=False, na=False)]

        with resultado_contenedor.container():
            if not productos_filtrados.empty:
                st.write(f"Productos que coinciden con '{nombre_del_producto}':")
                for _, row in productos_filtrados.iterrows():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        # Verificar si existe un precio_actual, sino usar precio_descuento
                        precio = row['precio_actual'] if pd.notna(row['precio_actual']) else row['precio_descuento']
                        
                        # Crear el enlace usando markdown, que abre en una nueva pestaña
                        st.markdown(f"[**{row['nombre']}**]({row['link']})", unsafe_allow_html=True)
                        
                        st.write(f"Precio: {precio}")
                        st.write(f"Tienda: {row['supermarket']}")
                        st.write(f"Marca: {row['marca']}")
                    with col2:
                        
                       st.markdown(f"[![Imagen]({row['imagen_url']})]({row['link']})", unsafe_allow_html=True)
                       
                    st.write("---")
            else:
                st.write("No se encontraron productos que coincidan con la búsqueda.")

