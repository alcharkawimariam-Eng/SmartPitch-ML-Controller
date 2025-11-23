# SmartPitch – ML-Based Region 3 Pitch Controller

SmartPitch is a data-driven pitch angle assistant for utility-scale wind turbines operating in **Region 3** (above rated wind speed).
It predicts the required blade pitch angle from:

* Wind speed (m/s)
* Rotor speed (RPM)
* Active power (kW)

and returns:

* **Raw pitch** (unconstrained ML output)
* **Safe pitch** (clipped to the physical range 0°–30°)

This makes SmartPitch suitable as a **research tool**, a **controller design assistant**, and an **educational simulator** for high-wind operation.

---

## 1. Background: Wind Turbines and Region 3

### 1.1 What is a wind turbine?

A horizontal-axis wind turbine consists of:

* **Rotor blades** that capture wind energy
* **Hub and nacelle** housing drivetrain and generator
* **Tower** providing structural support
* **Pitch system** that rotates the blades around their axes

The **pitch angle** controls how much aerodynamic force each blade captures:

* Small pitch → more lift → more aerodynamic torque
* Large pitch → less lift → reduced torque
* Very large pitch → used for braking and shutdown

### 1.2 Wind turbine operating regions

Wind turbines are typically controlled in three operating regions:

**Region 1 – Below cut-in wind speed**
The wind is too weak; the turbine remains idle.

**Region 2 – Between cut-in and rated wind speed**
Goal: maximise energy capture.
The pitch stays small; control focuses on torque or generator speed.

**Region 3 – Above rated wind speed (our focus)**
The incoming wind power is too high.
Goal: **keep generator power near rated** and **protect the turbine**.
Pitch control becomes the dominant mechanism.

In Region 3:

* Aerodynamic forces are large
* Blade pitch is increased to reduce lift
* Rotor overspeed and overload must be prevented
* Effective pitch control is critical for safety

---

## 2. Why Machine Learning?

Traditional pitch controllers rely on:

* Lookup tables
* Gain-scheduled PID controllers
* Linear approximations

They may struggle when:

* Turbulence intensity changes
* Atmospheric conditions drift
* The turbine is derated
* Components age or degrade

Machine learning can:

* Learn nonlinear relationships directly from SCADA data
* Adapt when retrained
* Provide transparent raw vs. clipped outputs
* Enable fast what-if analysis and testing

SmartPitch is not a certified controller but a **decision-support tool** for research.

---

## 3. Machine Learning Model

* Model: `MLPRegressor` (fully-connected neural network)
* Inputs:

  * wind_speed (m/s)
  * rotor_speed (RPM)
  * power (kW)
* Target:

  * pitch angle (deg)

### Preprocessing

* Region 3 filtering
* Outlier removal
* Normalization using `StandardScaler`
* Dataset stored in: `data/region3_clean.csv`

### Model performance (test set)

* MAE ≈ 0.55°
* R² ≈ 0.95

### Saved ML artifacts

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

### Run manually

```
uvicorn api.main:app --reload
```

### Endpoints

**GET /health**
Health check.

**POST /predict**
Input:

```json
{
  "wind_speed": 15.0,
  "rotor_speed": 12.0,
  "power": 3600
}
```

Output:

```json
{
  "pitch": 28.5,
  "pitch_raw": 34.2
}
```

* `pitch_raw`: raw ML output
* `pitch`: clipped into `[0°, 30°]` for safety

---

## 6. Streamlit UI

### Run manually

```
streamlit run app/frontend/streamlit_app.py
```

### Features

* Real-time pitch prediction
* Displays raw vs. safe pitch
* Automatic warnings when clipping occurs
* Random SCADA sample loading
* Comparison with true pitch
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

* API: [http://localhost:8000](http://localhost:8000)
* API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
* UI: [http://localhost:8501](http://localhost:8501)

---

## 8. Benefits and Use Cases

SmartPitch allows:

* Studying Region 3 pitch behaviour
* Inspecting extreme operating points
* Understanding when and why pitch saturates
* Supporting hybrid controller development
* Teaching and demonstrating wind turbine control concepts

---

## 9. Planned Enhancements

* Pitch vs. wind-speed “what-if” curve
* Time-series evaluation from CSV upload
* Power derating modes
* Extreme-load warnings
* Switchable models (MLP, RF, SVR)
* API security tokens
* Advanced Region 3 visualisation

---

## 10. Team

* Mariam Charkawi
* Joumana Saker
* Supervisor: Dr Ammar Mohanna
* Faculty of Engineering & Architecture – AUB
