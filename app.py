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

# Crear una columna que combine nombre y marca
data['nombre_marca'] = data['nombre'].astype(str) + " " + data.get('marca', '')

# Reemplazar valores NaN en la columna combinada
data['nombre_marca'] = data['nombre_marca'].fillna('').str.lower()

# Función para filtrar productos
def buscar_productos(nombre_busqueda, data):
    # Convertir la búsqueda a minúsculas y dividirla en palabras
    palabras = nombre_busqueda.lower().split()

    # Crear expresión regular para buscar todas las palabras, en cualquier orden
    regex = ".*".join(palabras)  # Combina las palabras con .* (cualquier cosa en medio)

    # Filtrar productos que contengan las palabras en 'nombre_marca'
    productos_filtrados = data[data['nombre_marca'].str.contains(regex, na=False, regex=True)]
    return productos_filtrados

# Interfaz principal
st.title('Consulta de Recomendación de Productos')
st.write('Busque productos por nombre o marca para encontrar recomendaciones.')

# Campo de entrada de búsqueda
nombre_busqueda = st.text_input(
    'Escriba el nombre del producto o la marca:',
    value='',
    placeholder='Ejemplo: zapatillas converse'
)

# Botón de búsqueda
if st.button('Buscar'):
    if nombre_busqueda.strip():
        resultados = buscar_productos(nombre_busqueda, data)
        if not resultados.empty:
            st.write(f"Resultados para '{nombre_busqueda}':")
            for _, row in resultados.iterrows():
                col1, col2 = st.columns([3, 1])
                with col1:
                    precio = row['precio_actual'] if pd.notna(row['precio_actual']) else row['precio_descuento']
                    st.markdown(f"[**{row['nombre']}**]({row['link']})", unsafe_allow_html=True)
                    st.write(f"Marca: {row.get('marca', 'Sin marca')}")
                    st.write(f"Precio: {precio}")
                    st.write(f"Tienda: {row['supermarket']}")
                with col2:
                    if pd.notna(row['imagen_url']):
                        st.markdown(f"[![Imagen]({row['imagen_url']})]({row['link']})", unsafe_allow_html=True)
                st.write("---")
        else:
            st.write(f"No se encontraron productos que coincidan con '{nombre_busqueda}'.")
    else:
        st.write("Por favor, ingrese un término de búsqueda.")