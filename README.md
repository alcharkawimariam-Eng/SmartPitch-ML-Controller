# SmartPitch – ML-Based Region 3 Pitch Controller

SmartPitch is a data-driven pitch angle assistant for utility-scale wind turbines operating in **Region 3** (above rated wind speed).
It predicts the optimal pitch angle based on SCADA measurements:

* Wind speed (m/s)
* Rotor speed (RPM)
* Active power (kW)

The system outputs:

* **Raw pitch** (unconstrained ML output)
* **Safe pitch** (clipped to the physical range **0°–30°**)

This makes SmartPitch suitable as a **research tool**, a **controller-design assistant**, and an **educational simulator** for Region 3 dynamics.

---

## 1. Background: Wind Turbines & Region 3

Modern horizontal-axis wind turbines operate in three regions:

### Region 1

Below cut-in wind speed. Turbine is idle.

### Region 2

Between cut-in and rated wind speed.
Goal: maximize energy capture (MPPT).
Pitch angle remains small.

### Region 3

Wind is above rated.
Goal: **keep generator power near rated** and **protect the turbine**.
Pitch angle increases to shed aerodynamic loads.

Because Region 3 involves strong and turbulent winds, effective pitch control is critical to:

* limit torque
* avoid overspeed
* reduce mechanical stress
* maintain safe operation

---

## 2. Why Machine Learning?

Traditional strategies rely on:

* lookup tables
* PID or gain-scheduled controllers
* simplified aerodynamic models

These can struggle when conditions vary (turbulence, aging, derating, noise).

Machine Learning offers:

* ability to learn nonlinear relationships directly from SCADA data
* adaptive behavior when retrained
* fast inference suitable for interactive analysis
* transparent raw vs. safe predictions

SmartPitch is not intended as a certified controller—it is a **decision-support tool**.

---

## 3. Machine Learning Model

* Model: `MLPRegressor` (scikit-learn)
* Inputs: wind speed, rotor speed, power
* Target: pitch angle
* Scaling: `StandardScaler`
* Dataset: cleaned Region 3 SCADA (`data/region3_clean.csv`)

### Model performance (test set)

* MAE ≈ 0.55°
* R² ≈ 0.95

Saved artifacts:

```
api/models/standard_scaler.joblib
api/models/mlp_pitch_regressor_best.joblib
```

---

## 4. Project Structure

```
SmartPitch-ML-Controller/
│
├── api/
│   ├── main.py
│   ├── model_loader.py
│   ├── schemas.py
│   └── models/
│
├── app/
│   └── frontend/
│       └── streamlit_app.py
│
├── notebooks/
├── data/
├── Dockerfile.api
├── Dockerfile.ui
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 5. FastAPI Backend

### Run without Docker

```
uvicorn api.main:app --reload
```

### Endpoints

**GET /health**
Health check.

**GET /model_info**
Returns model and feature information.

**POST /predict**
Input:

```json
{
  "wind_speed": 15.2,
  "rotor_speed": 12.1,
  "power": 3500
}
```

Output:

```json
{
  "pitch": 25.4,
  "pitch_raw": 32.8
}
```

`pitch` is clipped to `[0°, 30°]`.
`pitch_raw` exposes the unconstrained model output.

---

## 6. Streamlit User Interface

### Run without Docker

```
streamlit run app/frontend/streamlit_app.py
```

### Features

* Real-time pitch prediction
* Raw + safe pitch values
* Warnings when clipping occurs
* Random Region-3 SCADA sampling
* Comparison with true pitch values
* Scenario history
* Tabs:

  * SmartPitch Console
  * Project Overview
  * Region 3 Basics

---

## 7. Docker Deployment

### Build

```
docker compose build
```

### Run

```
docker compose up
```

### Access

* API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
* Streamlit UI: [http://localhost:8501](http://localhost:8501)

---

## 8. Use Cases & Benefits

SmartPitch can be used to:

* Study pitch behaviour in Region 3
* Validate extreme conditions and clipping
* Assist controller development
* Explore nonlinear SCADA relationships
* Provide educational insight into high-wind operation

---

## 9. Planned Enhancements

1. What-if pitch vs wind-speed curve
2. Time-series mode (CSV upload → full prediction)
3. Power derating scenarios (100%, 90%, 80%)
4. Extreme event warnings
5. Model selection (MLP, RF, SVR)
6. API authentication
7. Extended Region-3 visualisations

---

## 10. Team

* Mariam Charkawi
* Joumana Saker
* Supervisor: Dr Ammar Mohanna
* Faculty of Engineering & Architecture – AUB


Just ask!
