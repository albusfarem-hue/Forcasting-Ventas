# ==============================
# 🚀 INICIO REAL DE LA APP
# ==============================

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Rutas absolutas basadas en la ubicación real de app.py
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "modelo_final.joblib"
DATA_PATH  = BASE_DIR / "data" / "processed" / "inferencia_df_transformado.csv"

# Configuración de la página
st.set_page_config(
    page_title="Simulador de Ventas - Noviembre 2025",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("App de Machine Learning - Forecasting Ventas")
st.write("Bienvenido a la aplicación de predicción de ventas.")

# Diagnóstico rápido (opcional pero útil)
st.write("MODEL_PATH:", str(MODEL_PATH))
st.write("DATA_PATH:", str(DATA_PATH))

# Carga segura
modelo = joblib.load(MODEL_PATH)
inferencia_df = pd.read_csv(DATA_PATH)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stMetric label {
        color: #1e3a8a !important;
        font-weight: 600 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #1e40af !important;
        font-weight: 700 !important;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    div[data-testid="column"] > div > div > div > div {
        color: #1e40af !important;
    }
    h3 {
        color: #1e3a8a !important;
    }
    </style>
""", unsafe_allow_html=True)

# Cargar modelo y datos
@st.cache_resource
def cargar_modelo():
    try:
        modelo = joblib.load('../models/modelo_final.joblib')
        return modelo
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        return None

@st.cache_data
def cargar_datos():
    try:
        df = pd.read_csv('../data/processed/inferencia_df_transformado.csv')
        df['fecha'] = pd.to_datetime(df['fecha'])
        
        # Calcular precio_competencia si no existe
        if 'precio_competencia' not in df.columns:
            if all(col in df.columns for col in ['Amazon', 'Decathlon', 'Deporvillage']):
                df['precio_competencia'] = (df['Amazon'] + df['Decathlon'] + df['Deporvillage']) / 3
        
        return df
    except Exception as e:
        st.error(f"❌ Error al cargar los datos: {e}")
        return None

# Función para actualizar variables dependientes
def actualizar_variables(df_producto, descuento_pct, escenario_competencia):
    df_sim = df_producto.copy()
    
    # Actualizar precio_venta según descuento
    df_sim['precio_venta'] = df_sim['precio_base'] * (1 + descuento_pct / 100)
    
    # Actualizar precios de competencia según escenario
    if escenario_competencia == "Competencia -5%":
        factor = 0.95
    elif escenario_competencia == "Competencia +5%":
        factor = 1.05
    else:
        factor = 1.0
    
    df_sim['Amazon'] = df_sim['Amazon'] * factor
    df_sim['Decathlon'] = df_sim['Decathlon'] * factor
    df_sim['Deporvillage'] = df_sim['Deporvillage'] * factor
    
    # Recalcular precio_competencia y ratio_precio
    df_sim['precio_competencia'] = (df_sim['Amazon'] + df_sim['Decathlon'] + df_sim['Deporvillage']) / 3
    df_sim['ratio_precio'] = df_sim['precio_venta'] / df_sim['precio_competencia']
    
    # Calcular descuento_porcentaje
    df_sim['descuento_porcentaje'] = ((df_sim['precio_base'] - df_sim['precio_venta']) / df_sim['precio_base']) * 100
    
    return df_sim

# Función para hacer predicciones recursivas
def predecir_recursivo(df_sim, modelo, columnas_modelo):
    df_pred = df_sim.copy()
    df_pred = df_pred.sort_values('fecha').reset_index(drop=True)
    
    # Asegurar que todas las columnas que el modelo espera estén presentes
    columnas_faltantes = set(columnas_modelo) - set(df_pred.columns)
    for col in columnas_faltantes:
        df_pred[col] = 0
    
    predicciones = []
    
    for idx in range(len(df_pred)):
        # Preparar datos para predicción - asegurar el orden correcto
        X_pred = df_pred.loc[idx:idx, columnas_modelo]
        
        # Hacer predicción
        pred = modelo.predict(X_pred)[0]
        pred = max(0, pred)  # No permitir valores negativos
        predicciones.append(pred)
        
        # Actualizar lags para el siguiente día (si no es el último)
        if idx < len(df_pred) - 1:
            # Desplazar lags hacia la derecha
            for lag in range(7, 1, -1):
                col_actual = f'unidades_vendidas_lag{lag}'
                col_anterior = f'unidades_vendidas_lag{lag-1}'
                if col_actual in df_pred.columns and col_anterior in df_pred.columns:
                    df_pred.loc[idx + 1, col_actual] = df_pred.loc[idx, col_anterior]
            
            # Actualizar lag_1 con la predicción actual
            if 'unidades_vendidas_lag1' in df_pred.columns:
                df_pred.loc[idx + 1, 'unidades_vendidas_lag1'] = pred
            
            # Actualizar media móvil de 7 días
            if 'unidades_vendidas_mm7' in df_pred.columns:
                ultimas_7 = predicciones[-7:] if len(predicciones) >= 7 else predicciones
                df_pred.loc[idx + 1, 'unidades_vendidas_mm7'] = np.mean(ultimas_7)
    
    df_pred['unidades_predichas'] = predicciones
    df_pred['ingresos_proyectados'] = df_pred['unidades_predichas'] * df_pred['precio_venta']
    
    return df_pred

# Cargar modelo y datos
modelo = cargar_modelo()
df_completo = cargar_datos()

if modelo is None or df_completo is None:
    st.stop()

# Obtener columnas que el modelo espera
columnas_modelo = modelo.feature_names_in_

# Asegurar que el dataframe tenga todas las columnas necesarias
columnas_faltantes = set(columnas_modelo) - set(df_completo.columns)
for col in columnas_faltantes:
    df_completo[col] = 0

# SIDEBAR - Controles de simulación
st.sidebar.title("🎛️ Controles de Simulación")
st.sidebar.markdown("---")

# Selector de producto
productos_disponibles = sorted(df_completo['nombre'].unique())
producto_seleccionado = st.sidebar.selectbox(
    "📦 Seleccionar Producto",
    productos_disponibles,
    index=0
)

# Slider de descuento
descuento = st.sidebar.slider(
    "💰 Ajuste de Descuento (%)",
    min_value=-50,
    max_value=50,
    value=0,
    step=5,
    help="Ajusta el precio de venta del producto"
)

# Selector de escenario de competencia
st.sidebar.markdown("### 🏪 Escenario de Competencia")
escenario = st.sidebar.radio(
    "",
    ["Actual (0%)", "Competencia -5%", "Competencia +5%"],
    index=0,
    help="Simula cambios en los precios de la competencia"
)

# Botón de simulación
simular = st.sidebar.button("🚀 Simular Ventas", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("💡 Ajusta los controles y presiona 'Simular Ventas' para ver las predicciones actualizadas.")

# ZONA PRINCIPAL
st.title("📊 Dashboard de Simulación de Ventas - Noviembre 2025")
st.markdown(f"### Producto: **{producto_seleccionado}**")
st.markdown("---")

# Ejecutar simulación
if simular:
    with st.spinner("🔄 Procesando predicciones recursivas..."):
        # Filtrar datos del producto seleccionado
        df_producto = df_completo[df_completo['nombre'] == producto_seleccionado].copy()
        
        if len(df_producto) == 0:
            st.error("❌ No se encontraron datos para el producto seleccionado.")
            st.stop()
        
        # Actualizar variables según controles
        df_simulacion = actualizar_variables(df_producto, descuento, escenario)
        
        # Hacer predicciones recursivas
        df_resultados = predecir_recursivo(df_simulacion, modelo, columnas_modelo)
        
        # Calcular KPIs
        unidades_totales = df_resultados['unidades_predichas'].sum()
        ingresos_totales = df_resultados['ingresos_proyectados'].sum()
        precio_promedio = df_resultados['precio_venta'].mean()
        descuento_promedio = df_resultados['descuento_porcentaje'].mean()
        
        # Mostrar KPIs
        st.markdown("## 📈 Indicadores Clave")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🛒 Unidades Totales",
                value=f"{int(unidades_totales):,}",
                delta=None
            )
        
        with col2:
            st.metric(
                label="💶 Ingresos Totales",
                value=f"€{ingresos_totales:,.2f}",
                delta=None
            )
        
        with col3:
            st.metric(
                label="💵 Precio Promedio",
                value=f"€{precio_promedio:.2f}",
                delta=None
            )
        
        with col4:
            st.metric(
                label="🏷️ Descuento Promedio",
                value=f"{descuento_promedio:.1f}%",
                delta=None
            )
        
        st.markdown("---")
        
        # Gráfico de predicción diaria
        st.markdown("## 📅 Predicción de Ventas Diarias")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.set_style("whitegrid")
        
        # Gráfico de línea principal
        dias = df_resultados['dia_mes'].values
        ventas = df_resultados['unidades_predichas'].values
        
        ax.plot(dias, ventas, marker='o', linewidth=2.5, markersize=6, 
                color='#667eea', label='Ventas Predichas')
        
        # Marcar Black Friday (día 28)
        bf_idx = df_resultados[df_resultados['dia_mes'] == 28].index
        if len(bf_idx) > 0:
            bf_dia = 28
            bf_ventas = df_resultados.loc[bf_idx[0], 'unidades_predichas']
            
            ax.axvline(x=bf_dia, color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax.plot(bf_dia, bf_ventas, 'ro', markersize=12, zorder=5)
            ax.annotate('🛍️ Black Friday', 
                       xy=(bf_dia, bf_ventas),
                       xytext=(bf_dia - 3, bf_ventas * 1.15),
                       fontsize=12,
                       fontweight='bold',
                       color='red',
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        ax.set_xlabel('Día del Mes', fontsize=12, fontweight='bold')
        ax.set_ylabel('Unidades Vendidas', fontsize=12, fontweight='bold')
        ax.set_title(f'Ventas Predichas - Noviembre 2025', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, 31))
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Tabla detallada
        st.markdown("## 📋 Detalle de Predicciones")
        
        # Preparar tabla para mostrar
        df_tabla = df_resultados[['fecha', 'nombre_dia_semana', 'precio_venta', 
                                  'precio_competencia', 'descuento_porcentaje',
                                  'unidades_predichas', 'ingresos_proyectados']].copy()
        
        df_tabla['fecha'] = df_tabla['fecha'].dt.strftime('%Y-%m-%d')
        df_tabla['precio_venta'] = df_tabla['precio_venta'].apply(lambda x: f"€{x:.2f}")
        df_tabla['precio_competencia'] = df_tabla['precio_competencia'].apply(lambda x: f"€{x:.2f}")
        df_tabla['descuento_porcentaje'] = df_tabla['descuento_porcentaje'].apply(lambda x: f"{x:.1f}%")
        df_tabla['unidades_predichas'] = df_tabla['unidades_predichas'].apply(lambda x: f"{int(x):,}")
        df_tabla['ingresos_proyectados'] = df_tabla['ingresos_proyectados'].apply(lambda x: f"€{x:.2f}")
        
        df_tabla.columns = ['Fecha', 'Día', 'Precio Venta', 'Precio Competencia', 
                           'Descuento', 'Unidades', 'Ingresos']
        
        # Destacar Black Friday
        def highlight_bf(row):
            if '28' in row['Fecha']:
                return ['background-color: #ffcccc'] * len(row)
            return [''] * len(row)
        
        st.dataframe(
            df_tabla.style.apply(highlight_bf, axis=1),
            use_container_width=True,
            height=400
        )
        
        st.markdown("---")
        
        # Comparativa de escenarios
        st.markdown("## 🔄 Comparativa de Escenarios de Competencia")
        st.markdown("*Manteniendo el descuento actual, variando solo precios de competencia*")
        
        escenarios = ["Actual (0%)", "Competencia -5%", "Competencia +5%"]
        resultados_escenarios = {}
        
        with st.spinner("📊 Calculando escenarios alternativos..."):
            for esc in escenarios:
                df_esc = actualizar_variables(df_producto, descuento, esc)
                df_esc_pred = predecir_recursivo(df_esc, modelo, columnas_modelo)
                resultados_escenarios[esc] = {
                    'unidades': df_esc_pred['unidades_predichas'].sum(),
                    'ingresos': df_esc_pred['ingresos_proyectados'].sum()
                }
        
        col1, col2, col3 = st.columns(3)
        
        for idx, (esc, resultados) in enumerate(resultados_escenarios.items()):
            col = [col1, col2, col3][idx]
            with col:
                st.markdown(f"""
                    <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 10px;">
                        <h3 style="color: #1e3a8a; text-align: center; margin-bottom: 15px;">{esc}</h3>
                    </div>
                """, unsafe_allow_html=True)
                st.metric(
                    label="🛒 Unidades",
                    value=f"{int(resultados['unidades']):,}"
                )
                st.metric(
                    label="💶 Ingresos",
                    value=f"€{resultados['ingresos']:,.2f}"
                )
        
        st.success("✅ Simulación completada exitosamente!")

else:
    st.info("👈 Por favor, configura los parámetros en el panel lateral y presiona 'Simular Ventas'.")
    
    # Mostrar información inicial del producto seleccionado
    df_info = df_completo[df_completo['nombre'] == producto_seleccionado].iloc[0]
    
    st.markdown("## 📦 Información del Producto")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Categoría:** {df_info['categoria']}")
        st.markdown(f"**Subcategoría:** {df_info['subcategoria']}")
        st.markdown(f"**Precio Base:** €{df_info['precio_base']:.2f}")
    
    with col2:
        st.markdown(f"**Producto Estrella:** {'✅ Sí' if df_info['es_estrella'] else '❌ No'}")
        if 'precio_competencia' in df_info:
            st.markdown(f"**Precio Competencia Actual:** €{df_info['precio_competencia']:.2f}")
        elif all(col in df_info for col in ['Amazon', 'Decathlon', 'Deporvillage']):
            precio_comp = (df_info['Amazon'] + df_info['Decathlon'] + df_info['Deporvillage']) / 3
            st.markdown(f"**Precio Competencia Actual:** €{precio_comp:.2f}")
