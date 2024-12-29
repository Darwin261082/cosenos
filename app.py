import streamlit as st
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
data['nombre_marca'] = data['nombre_marca'].fillna('').str.lower()

# Conjunto para almacenar los índices de productos ya recomendados
productos_recomendados = set()

# Función para filtrar productos base según palabras de búsqueda
def buscar_productos(nombre_busqueda, data):
    palabras = nombre_busqueda.lower().split()
    regex = ".*".join(palabras)
    productos_filtrados = data[data['nombre_marca'].str.contains(regex, na=False, regex=True)]
    return productos_filtrados

# Función para calcular similitud
def calcular_similitud(producto_base_idx, data, productos_base_idx, umbral_similitud=0.22):
    df = data.copy()
    tfidf = TfidfVectorizer()
    tfidf_matriz = tfidf.fit_transform(df['nombre_marca'])
    similitudes = cosine_similarity(tfidf_matriz[producto_base_idx], tfidf_matriz).flatten()
    df['similitud'] = similitudes
    df = df[~df.index.isin(productos_base_idx)]  # Excluir productos base
    recomendaciones_filtradas = df[df['similitud'] >= umbral_similitud]  # Filtrar por umbral
    recomendaciones_filtradas = recomendaciones_filtradas[~recomendaciones_filtradas.index.isin(productos_recomendados)]  # Evitar duplicados
    recomendados = recomendaciones_filtradas.sort_values(by='similitud', ascending=False).head(5)
    return recomendados

# Interfaz principal
st.title('Recomendación de Productos por Similitud de Coseno')
st.write('Busque productos por nombre o marca para encontrar recomendaciones.')

# Entrada de búsqueda del producto base
nombre_busqueda = st.text_input(
    'Escriba el nombre del producto o la marca:',
    value='',
    placeholder='Ejemplo: zapatillas converse'
)

# Botón de búsqueda
if st.button('Buscar'):
    if nombre_busqueda.strip():
        productos_base = buscar_productos(nombre_busqueda, data)
        
        if not productos_base.empty:
            productos_base_idx = productos_base.index.tolist()
            st.write(f"**Se encontraron {len(productos_base)} productos base que coinciden con la búsqueda:**")
            
            for _, producto_base in productos_base.iterrows():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"[**{producto_base['nombre']}**]({producto_base['link']})", unsafe_allow_html=True)
                    st.write(f"Marca: {producto_base.get('marca', 'Sin marca')}")
                    st.write(f"Precio: {producto_base.get('precio_actual', 'No disponible')}")
                    st.write(f"Tienda: {producto_base.get('supermarket', 'No especificada')}")
                with col2:
                    if pd.notna(producto_base.get('imagen_url', None)):
                        st.markdown(
                            f"[![Imagen del producto]({producto_base['imagen_url']})]({producto_base['link']})",
                            unsafe_allow_html=True
                        )
                st.write("---")
            
            st.write("### Recomendaciones:")
            for producto_idx, producto_base in productos_base.iterrows():
                recomendaciones = calcular_similitud(producto_idx, data, productos_base_idx)
                if not recomendaciones.empty:
                    st.write(f"Productos similares a **{producto_base['nombre']}**:")
                    for _, row in recomendaciones.iterrows():
                        # Agregar índice del producto al conjunto de productos recomendados
                        productos_recomendados.add(row.name)
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"[**{row['nombre']}**]({row['link']})", unsafe_allow_html=True)
                            st.write(f"Marca: {row.get('marca', 'Sin marca')}")
                            st.write(f"Precio: {row['precio_actual'] if pd.notna(row['precio_actual']) else row['precio_descuento']}")
                            st.write(f"Tienda: {row['supermarket']}")
                        with col2:
                            if pd.notna(row['imagen_url']):
                                st.markdown(
                                    f"[![Imagen del producto]({row['imagen_url']})]({row['link']})",
                                    unsafe_allow_html=True
                                )
                        st.write("---")
                else:
                    st.write(f"No se encontraron productos similares a **{producto_base['nombre']}**.")
        else:
            st.write(f"No se encontraron productos que coincidan con '{nombre_busqueda}'.")
    else:
        st.write("Por favor, ingrese un término de búsqueda.")

