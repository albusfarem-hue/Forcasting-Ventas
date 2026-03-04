"""
Script para regenerar el modelo con cloudpickle (más robusto que pickle)
"""
import pandas as pd
import numpy as np
import cloudpickle
from sklearn.ensemble import HistGradientBoostingRegressor
from pathlib import Path
import warnings
import json
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "df.csv"
MODEL_PATH = BASE_DIR / "models" / "modelo_final.pkl"
METADATA_PATH = BASE_DIR / "models" / "modelo_metadata.json"

print("[*] Cargando datos...")
df = pd.read_csv(DATA_PATH)
print(f"[OK] Datos cargados: {df.shape}")

exclude_cols = ['fecha', 'unidades_vendidas', 'producto_id', 'nombre', 'categoria', 'subcategoria', 'nombre_dia_semana', 'dia_semana_es']
X_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype != 'object']
X = df[X_cols].astype(np.float32)
y = df['unidades_vendidas'].astype(np.float32)

print(f"[OK] Features: {X.shape}, {len(X_cols)} columnas")
print(f"[OK] Target: {y.shape}")

print("[*] Entrenando modelo...")
np.random.seed(42)
model = HistGradientBoostingRegressor(
    max_iter=50,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    warm_start=False
)

model.fit(X, y)
print("[OK] Modelo entrenado")

print(f"[*] Guardando con cloudpickle en: {MODEL_PATH}")
with open(MODEL_PATH, 'wb') as f:
    cloudpickle.dump(model, f)
print("[OK] Modelo guardado")

# Guardar metadatos
metadata = {
    'feature_names': X_cols,
    'n_features': len(X_cols),
    'modelo_type': 'HistGradientBoostingRegressor'
}
with open(METADATA_PATH, 'w') as f:
    json.dump(metadata, f)
print("[OK] Metadatos guardados")

# Verificar carga
print("[*] Verificando carga...")
with open(MODEL_PATH, 'rb') as f:
    test_model = cloudpickle.load(f)
print("[OK] Modelo cargado correctamente")

print("[OK] Completado!")


