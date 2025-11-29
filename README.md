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
---

## 🔹 New Update: NREL 5MW Dataset Integration

SmartPitch has been expanded to support a second major dataset: the **NREL 5MW Reference Wind Turbine**.  
Two new types of Region 3 data were added:

1. **Region 3 – Steady Operating Point Dataset**  
2. **Region 3 – Turbulent Wind-Profile Dataset**

Both datasets were fully cleaned, filtered, and analyzed using new EDA notebooks.  
This enhancement allows SmartPitch to study realistic turbulent high-wind conditions and compare model behavior across different inflow types.

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

## 🔹 Additional Machine Learning Models (NREL Dataset)

New ML workflows were added using the NREL 5MW datasets, with two training paths:

### **A) Region 3 – Operating Point Dataset**
Notebook: `05-region3-operating-point.ipynb`

- Clean Region 3 steady dataset  
- EDA + filtering  
- Models tested: MLP, Random Forest, Linear Regression  
- **Best model:** MLP (similar behavior to original SCADA model)

### **B) Region 3 – Turbulent Wind-Profile Dataset**
Notebook: `06-region3-wind-profile.ipynb`

- Time-varying turbulent inflow  
- Region 3 filtering  
- Models tested: MLP, Linear Regression, SVR, Random Forest  
- **Best model:** Random Forest  
  (most stable under turbulence and nonlinear behavior)

Two new datasets were added:


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

## 5. Quick Start
This section explains exactly how to run the project, including the FastAPI backend and the Streamlit UI.


### Create and activate a Virtual Environment (Windows)
Since we are running a machine-learning project, it's safer to create virtual environment This avoids version conflicts (especially with scikit-learn and numpy). from root path, create a virtual env

```
# Navigate to the project folder
cd SmartPitch-ML-Controller-main

# Create virtual environment
python -m venv venv


Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Activate it
.\venv\Scripts\Activate.ps1

```

### Install dependencies
```
pip install -r requirements.txt
```
If your environment complains about missing dependencies, you can install missing ones manually:

```
pip install uvicorn fastapi streamlit scikit-learn pandas matplotlib joblib
```

### Start the FastAPI backend
```
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
Keep this terminal open.

### Start the Streamlit UI
Open a new terminal and activate the venv again:
```
# cd to the root path
cd SmartPitch-ML-Controller-main
venv\Scripts\activate

streamlit run app/frontend/streamlit_app.py
```

---

## 6. FastAPI Backend

### Run manually

```
uvicorn api.main:app --reload
```

### Endpoints

**GET /health**
Health check.

**POST /predict**
returns raw and safe pitch
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

## 7. Streamlit UI

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

## 8. Docker Deployment

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
## 9. Project UI Overview
Below is the main interface of SmartPitch:

SmartPitch Landing Page
<img width="1863" height="943" alt="UI1" src="https://github.com/user-attachments/assets/d7d95163-8364-4167-b2c6-7d5b1ed1e1bf" />

SmartPitch Console – Real-time Pitch Prediction
 
<img width="1838" height="815" alt="UI2" src="https://github.com/user-attachments/assets/fd4da056-67bb-46a3-972c-9821983795ce" />

Scenario History

<img width="1868" height="470" alt="UI3" src="https://github.com/user-attachments/assets/c0d6920e-90ab-443f-b4c8-fd3e8a7cacfe" />

Project Overview Tab
<img width="1888" height="846" alt="UI4" src="https://github.com/user-attachments/assets/d64e0b5d-d50e-4518-9553-98fbd0c8cab2" />

Region 3 Basics

<img width="1865" height="781" alt="UI5" src="https://github.com/user-attachments/assets/206a40fb-9033-4f7b-8729-3aa087572c5e" />

What-if Lab

<img width="1832" height="882" alt="UI6" src="https://github.com/user-attachments/assets/dcebb450-c6c0-487b-9b19-e92154fae1b4" />
<img width="1831" height="582" alt="UI7" src="https://github.com/user-attachments/assets/97a78200-029f-4ac9-9a5d-51cf1ef612d7" />
<img width="1837" height="591" alt="UI8" src="https://github.com/user-attachments/assets/45e20cc7-2667-479d-8d38-3e165a2d3fee" />


Wind-Profile Model (Random Forest)
<img width="1825" height="671" alt="UI9" src="https://github.com/user-attachments/assets/1976bef7-0526-42d5-a01f-d0cf8ca89192" />


Wind-Profile Time Series
<img width="1857" height="881" alt="UI10" src="https://github.com/user-attachments/assets/ab2bd96e-fc75-46bb-93ab-2bfe6762990f" />

<img width="1842" height="478" alt="UI11" src="https://github.com/user-attachments/assets/d0644b2d-a5e1-472b-8a35-4b73d5257ec2" />


---
## 10. Benefits and Use Cases

SmartPitch allows:

* Studying Region 3 pitch behaviour
* Inspecting extreme operating points
* Understanding when and why pitch saturates
* Supporting hybrid controller development
* Teaching and demonstrating wind turbine control concepts
* Ability to compare steady-wind vs turbulent-wind performance  
* Evaluate model robustness under realistic NREL turbulence  
* Support for two parallel Region-3 modelling paths (operating-point vs wind-profile)

---

## 11. Planned Enhancements

* Power derating modes
* Extreme-load warnings
* Switchable models (MLP, RF, SVR)
* API security tokens
* Advanced Region 3 visualization

---

## 12. Team

* Mariam Charkawi
* Joumana Saker
* Supervisor: Dr Ammar Mohanna
* Faculty of Engineering & Architecture – AUB
