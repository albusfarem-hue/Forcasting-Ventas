"""
Script para regenerar el modelo sin problemas de compatibilidad NumPy
"""
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.ensemble import HistGradientBoostingRegressor
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "df.csv"
MODEL_PATH = BASE_DIR / "models" / "modelo_final.pkl"

print("[*] Cargando datos...")
df = pd.read_csv(DATA_PATH)
print(f"[OK] Datos cargados: {df.shape}")

# Excluir columnas no numéricas
exclude_cols = ['fecha', 'unidades_vendidas', 'producto_id', 'nombre', 'categoria', 'subcategoria', 'nombre_dia_semana', 'dia_semana_es']
X_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype != 'object']
X = df[X_cols].astype(np.float32)
y = df['unidades_vendidas'].astype(np.float32)

print(f"[OK] Features: {X.shape}")
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

# Guardar con pickle en lugar de joblib para evitar problemas de NumPy
print(f"[*] Guardando modelo en: {MODEL_PATH}")
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f, protocol=4)
print("[OK] Modelo guardado")

# Verificar que se puede cargar
print("[*] Verificando carga...")
with open(MODEL_PATH, 'rb') as f:
    test_model = pickle.load(f)
print("[OK] Modelo cargado exitosamente")
print("[OK] Completado!")
