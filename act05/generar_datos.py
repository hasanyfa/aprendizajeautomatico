import pandas as pd
import numpy as np

# Configurar semilla para reproducibilidad
np.random.seed(42)
n = 1000  # 1000 filas para tener suficientes datos

# Generar datos más realistas
regions = ['Norte', 'Centro', 'Sur', 'Occidente']

# Variables independientes
tenure_meses = np.random.gamma(2, 8, n).astype(int).clip(1, 48)  # Distribución más realista
tarifa_mensual = np.random.normal(169, 35, n).clip(80, 280)
horas_uso_semana = np.random.gamma(2, 4, n).clip(0.5, 35)
dispositivos_vinculados = np.random.poisson(2.5, n).clip(1, 8)
tickets_soporte_90d = np.random.poisson(1.2, n).clip(0, 8)
autopago = np.random.choice([0, 1], n, p=[0.35, 0.65])
recibio_promo = np.random.choice([0, 1], n, p=[0.75, 0.25])
region = np.random.choice(regions, n, p=[0.25, 0.35, 0.25, 0.15])

# Crear DataFrame
df = pd.DataFrame({
    'tenure_meses': tenure_meses,
    'tarifa_mensual': tarifa_mensual,
    'horas_uso_semana': horas_uso_semana,
    'dispositivos_vinculados': dispositivos_vinculados,
    'tickets_soporte_90d': tickets_soporte_90d,
    'autopago': autopago,
    'recibio_promo': recibio_promo,
    'region': region
})

# Generar variable dependiente (churn) de forma más realista
# Factores que aumentan el churn:
z = (-1.5  # Intercepto base
     + (-0.08) * df['tenure_meses']  # Menos antigüedad = más churn
     + (0.015) * (df['tarifa_mensual'] - df['tarifa_mensual'].mean())  # Tarifa alta = más churn
     + (-0.12) * df['horas_uso_semana']  # Menos uso = más churn
     + (-0.15) * df['dispositivos_vinculados']  # Menos dispositivos = más churn
     + (0.35) * df['tickets_soporte_90d']  # Más tickets = más churn
     + (-0.8) * df['autopago']  # Sin autopago = más churn
     + (-0.6) * df['recibio_promo']  # Sin promo = más churn
     )

# Efectos por región
region_effects = {'Norte': 0.0, 'Centro': 0.2, 'Sur': 0.4, 'Occidente': -0.3}
z += df['region'].map(region_effects)

# Añadir ruido para hacer más realista
z += np.random.normal(0, 0.3, n)

# Convertir a probabilidades usando función logística
prob_churn = 1 / (1 + np.exp(-z))

# Generar variable binaria de churn
churn = np.random.binomial(1, prob_churn, n)
df['churn'] = churn

# Guardar el archivo
df.to_csv('datos_churn.csv', index=False)

print(f"Dataset creado con {len(df)} filas")
print(f"Distribución de churn:")
print(df['churn'].value_counts())
print(f"Proporción de churn: {df['churn'].mean():.3f}")
