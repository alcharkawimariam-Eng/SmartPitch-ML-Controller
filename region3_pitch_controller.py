import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Load data
df = pd.read_csv("data/raw_scada.csv")

# Rename columns if needed (adjust to match your real ones)
df.rename(columns={
    "WindSpeed": "wind_speed",
    "Power_kW": "power",
    "RotorSpeed_RPM": "rotor_speed",
    "PitchAngle_deg": "pitch",
    "Timestamp": "timestamp"
}, inplace=True)

# Drop missing
df.dropna(subset=["wind_speed", "power", "rotor_speed", "pitch"], inplace=True)

# Filter Region 3 (wind >= 11 m/s)
df = df[df["wind_speed"] >= 11]

# Add previous pitch
df["pitch_prev"] = df["pitch"].shift(1)
df.dropna(subset=["pitch_prev"], inplace=True)

# Features & target
X = df[["wind_speed", "rotor_speed", "power", "pitch_prev"]]
y = df["pitch"]

# Sort by timestamp if exists
if "timestamp" in df.columns:
    df = df.sort_values("timestamp")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/feature_scaler.joblib")

# Train model
model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model
joblib.dump(model, "models/ml_pitch_controller.joblib")

# Evaluate
preds = model.predict(X_test_scaled)
print("MAE:", mean_absolute_error(y_test, preds))
print("RMSE:", mean_squared_error(y_test, preds, squared=False))
print("R²:", r2_score(y_test, preds))
