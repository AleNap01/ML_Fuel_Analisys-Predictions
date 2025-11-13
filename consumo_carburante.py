import psycopg2
import pandas as pd
import warnings
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import numpy as np

#  --- Configurazione ---
warnings.filterwarnings("ignore")

# --- Funzione per valutare i modelli---- 
results = {}
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"mse": mse, "r2": r2, "model": model}
    return model


# --- 1. Connessione al Database ---

conn = psycopg2.connect(
    dbname="provadin",
    user="postgres",
    password="Terzigno",
    host="localhost",
    port="5432"
)

query = """
SELECT * FROM telemetry_temp
WHERE
    ident = '352625697549655'
    AND can_fuel_consumed IS NOT NULL
    AND din_1 = true 
    AND engine_ignition_status = true
    AND position_speed > 0
    AND movement_status = true
    AND can_engine_rpm BETWEEN 1 AND 8000
    AND position_latitude BETWEEN -90 AND 90
    AND position_longitude BETWEEN -180 AND 180
    AND position_altitude BETWEEN -500 AND 10000
    AND position_direction BETWEEN 1 AND 359
    AND position_satellites > 4
    AND can_vehicle_mileage < 2000000;
"""

df = pd.read_sql_query(query, conn)
conn.close()



# --- 2. Preprocessing e Feature Engineering ---

df = df.sort_values('timestamp')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')
df['fuel_delta'] = df['can_fuel_consumed'].diff()
df = df[df['fuel_delta'] > 0]  
df['fuel_delta_smooth'] = df['fuel_delta'].ewm(span=20, adjust=False).mean()

feature_cols = [
    'battery_voltage',
    'can_engine_rpm',
    'can_pedal_brake_status',
    'can_throttle_pedal_level',
    'position_altitude',
    'position_direction',
    'position_speed',
    'vehicle_mileage',
    #'tipo_strada',
    #'limite_velocità',
    #'velocità_vs_limite'
]
target_col = 'fuel_delta_smooth'

df['veicle_mileage'] = np.sqrt(df['vehicle_mileage']) * 0.01

df_model = df[feature_cols + [target_col]].dropna()
X = df_model[feature_cols]
y = df_model[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Addestramento e Valutazione ---
 
# A. Random Forest
evaluate_model("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), X_train, y_train, X_test, y_test)

# B. Gradient Boosting
evaluate_model("Gradient Boosting", HistGradientBoostingRegressor(random_state=42), X_train, y_train, X_test, y_test)

# C. Linear Regression
evaluate_model("Linear Regression", LinearRegression(), X_train, y_train, X_test, y_test)

# D. XGBoost
evaluate_model("XGBoost", XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1), X_train, y_train, X_test, y_test)


print("\nRiepilogo finale modelli:")
for name, metrics in results.items():
    print(f"{name:20}  MSE: {metrics['mse']:.4f}  R²: {metrics['r2']:.4f}")


# --- 4. Grafici e Visualizzazione--- 

#--- consumo del carburante ---
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['fuel_delta'], label='fuel_delta (originale)', alpha=0.4)
plt.plot(df.index, df['fuel_delta_smooth'], label='fuel_delta_smooth', color='orange')
plt.xlabel("Timestamp")
plt.ylabel("Consumo carburante istantaneo")
plt.title("Confronto: Valori originali vs Smoothing (media mobile)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---  Confronto Modelli ---
models = list(results.keys())
mses = [results[m]['mse'] for m in models]
r2s  = [results[m]['r2'] for m in models]

plt.figure(figsize=(8, 4))
plt.bar(models, r2s, color='skyblue')
plt.title("Confronto R² (più alto è meglio)")
plt.ylabel("R²")
plt.ylim(min(r2s) - 0.05, max(r2s) + 0.05)
plt.axhline(0, color='gray', linestyle='--')
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.bar(models, mses, color='salmon')
plt.title("Confronto MSE (più basso è meglio)")
plt.ylabel("Mean Squared Error")
plt.ylim(0, max(mses) + 0.01)
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

# --- 5. Selezione e Salvataggio del Miglior Modello --- 

mse_vals = np.array([v["mse"] for v in results.values()]).reshape(-1, 1)
r2_vals  = np.array([v["r2"] for v in results.values()]).reshape(-1, 1)

mse_scaled = MinMaxScaler().fit_transform(mse_vals)
r2_scaled  = MinMaxScaler().fit_transform(r2_vals)

combined_scores = {
    name: r2_scaled[i][0] - mse_scaled[i][0]
    for i, name in enumerate(results.keys())
}

best_model_name = max(combined_scores, key=combined_scores.get)
best_model = results[best_model_name]["model"]

print(f"\n Modello migliore (bilanciato R²↑ e MSE↓): {best_model_name}")

joblib.dump(best_model, f"best_model_{best_model_name.replace(' ', '_').lower()}.pkl")
