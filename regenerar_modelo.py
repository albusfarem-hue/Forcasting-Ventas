"""
Script para regenerar el modelo con compatibilidad de versiones
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from pathlib import Path

# Configuración
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "df.csv"
MODEL_PATH = BASE_DIR / "models" / "modelo_final.joblib"

print("🔄 Regenerando modelo...")
print(f"Cargando datos desde: {DATA_PATH}")

# Cargar datos
df = pd.read_csv(DATA_PATH)
print(f"✅ Datos cargados: {df.shape}")

# Definir columnas de features (basadas en las del notebook)
X_cols = [col for col in df.columns if col not in ['fecha', 'unidades_vendidas']]
X = df[X_cols]
y = df['unidades_vendidas']

print(f"✅ Features preparadas: {X.shape}")
print(f"✅ Target preparado: {y.shape}")

# Entrenar modelo simple (puede ser más rápido que el original)
print("🤖 Entrenando modelo...")
model = HistGradientBoostingRegressor(
    max_iter=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    n_iter_no_change=10,
    validation_fraction=0.1
)

model.fit(X, y)
print("✅ Modelo entrenado")

# Guardar con protocolo compatible
print(f"💾 Guardando modelo en: {MODEL_PATH}")
joblib.dump(model, MODEL_PATH, protocol=4, compress=3)
print("✅ Modelo guardado exitosamente")

# Verificar que se puede cargar
print("🔍 Verificando que el modelo se puede cargar...")
modelo_prueba = joblib.load(MODEL_PATH)
print("✅ Modelo cargado correctamente")

print("\n¡✨ Regeneración completada!")
