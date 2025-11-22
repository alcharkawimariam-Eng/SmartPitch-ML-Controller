import os
import datetime
import requests

import pandas as pd
import streamlit as st

# 🔗 Backend API URL (change if needed)
API_URL = os.getenv("API_URL", "http://localhost:8000")


# ================== HELPERS ==================


def get_model_info():
    """Fetch model info from the FastAPI backend."""
    try:
        resp = requests.get(f"{API_URL}/model_info", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return None
    return None


def call_predict_api(wind_speed: float, rotor_speed: float, power: float):
    """Send features to the FastAPI /predict endpoint and return pitch + raw response."""
    payload = {
        "wind_speed": wind_speed,
        "rotor_speed": rotor_speed,
        "power": power,
    }
    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
    resp.raise_for_status()
    data = resp.json()
    pitch = data.get("pitch")  # backend returns {"pitch": value}
    return pitch, data


@st.cache_data
def load_region3_data():
    """Load the cleaned Region 3 dataset for sampling."""
    return pd.read_csv("data/region3_clean.csv")


# ================== PAGE CONFIG ==================

st.set_page_config(
    page_title="SmartPitch – ML-Based Pitch Controller",
    page_icon="🌀",
    layout="wide",
)

# ---- Session state init ----
if "history" not in st.session_state:
    st.session_state["history"] = []

if "wind_speed" not in st.session_state:
    st.session_state["wind_speed"] = 15.0

if "rotor_speed" not in st.session_state:
    st.session_state["rotor_speed"] = 12.0

if "power" not in st.session_state:
    st.session_state["power"] = 4000.0

if "true_pitch" not in st.session_state:
    st.session_state["true_pitch"] = None

# Try to load dataset (for random real sample)
try:
    df_data = load_region3_data()
except Exception:
    df_data = None

# ================== HEADER & BRANDING ==================

logo_path = "assets/aub_logo.png"
turbine_path = "assets/turbine_region3.jpg"

top_left, top_right = st.columns([3, 2])

with top_left:
    st.markdown("### Faculty of Engineering & Architecture – AUB")
    st.title("🌀 SmartPitch – ML-Based Pitch Controller")
    st.caption("Graduate project – Region 3 collective pitch control using supervised ML (MLPRegressor)")

    # Performance card
    st.info(
        "**Model performance (test set)**  \n"
        "- Mean Absolute Error (MAE) ≈ **0.55°**  \n"
        "- Coefficient of determination R² ≈ **0.95**",
        icon="✅",
    )

with top_right:
    col_logo, col_dev = st.columns([1, 1])
    with col_logo:
        try:
            st.image(logo_path, use_container_width=True)
        except Exception:
            st.markdown("**American University of Beirut**")
    with col_dev:
        st.markdown("#### Project Team")
        st.markdown("- Mariam Charkawi \n- Joumana Saker")
        st.markdown("#### Supervisor")
        st.markdown("- Dr. Ammar Mohanna")

# Turbine banner
turb_col1, turb_col2 = st.columns([2, 3])
with turb_col1:
    try:
        st.image(turbine_path, use_container_width=True, caption="Onshore wind turbine operating in Region 3")
    except Exception:
        st.markdown("*(Turbine illustration – add `assets/turbine_region3.jpg` for an image here.)*")
with turb_col2:
    with st.expander("ℹ️ **What is SmartPitch? (click to read)**", expanded=True):
        st.markdown(
            """
**SmartPitch** is an ML-based pitch controller for **Region 3** of a wind turbine.  
Given the measured **wind speed, rotor speed, and active power**, the model suggests the 
blade pitch angle that keeps the turbine near rated power while limiting mechanical loads.

The pipeline is:

1. **SCADA data** → filter Region 3.
2. **Supervised learning** (MLPRegressor) on `wind_speed`, `rotor_speed`, `power` → target `pitch`.
3. Export trained model → wrap in **FastAPI**.
4. Expose as `/predict` endpoint.
5. Build this **interactive Streamlit UI** for real-time what-if analysis.
"""
        )

st.markdown("---")

# ================== TABS ==================

tab_console, tab_overview, tab_region3 = st.tabs(
    ["🎛 SmartPitch Console", "📘 Project Overview", "🌬️ Region 3 Basics"]
)

# ========= TAB 1: CONSOLE =========
with tab_console:
    col_inputs, col_output = st.columns([1.1, 1.2])

    with col_inputs:
        st.subheader("Input Conditions")

        # ---- Defaults from session_state (if any) ----
        default_ws = float(st.session_state.get("wind_speed", 15.0))
        default_rs = float(st.session_state.get("rotor_speed", 12.0))
        default_p = float(st.session_state.get("power", 4000.0))

         # Remember previous values BEFORE sliders (to detect manual changes)
        prev_ws = default_ws
        prev_rs = default_rs
        prev_p  = default_p

        # sliders (no custom keys, no on_change)
        wind_speed = st.slider(
            "Wind Speed [m/s]",
            min_value=10.0,
            max_value=25.0,
            value=default_ws,
            step=0.1,
            help="Region 3 typically starts around 10–11 m/s.",
        )

        rotor_speed = st.slider(
            "Rotor Speed [RPM]",
            min_value=9.0,
            max_value=20.0,
            value=default_rs,
            step=0.1,
            help="Rotor speed in Region 3 (near rated).",
        )

        power = st.slider(
            "Active Power [kW]",
            min_value=0.0,
            max_value=5000.0,
            value=default_p,
            step=50.0,
            help="Generated electrical power.",
        )

        # keep current slider values in session_state
        st.session_state["wind_speed"] = wind_speed
        st.session_state["rotor_speed"] = rotor_speed
        st.session_state["power"] = power

         # ❗ If user changed any slider, forget the old true_pitch
        if (
            wind_speed != prev_ws
            or rotor_speed != prev_rs
            or power != prev_p
           ):
           st.session_state["true_pitch"] = None

        st.markdown("---")

        # 🎲 Use a random real SCADA sample
        if df_data is not None:
            if st.button("🎲 Use random real SCADA sample", use_container_width=True):
                sample = df_data.sample(1).iloc[0]

                st.session_state["wind_speed"] = float(sample["wind_speed"])
                st.session_state["rotor_speed"] = float(sample["rotor_speed"])
                st.session_state["power"] = float(sample["power"])
                st.session_state["true_pitch"] = float(sample["pitch"])

                # force full rerun so sliders jump to the sample values
                st.rerun()
        else:
            st.caption("Dataset not found (expected `data/region3_clean.csv`).")

        st.markdown("")
        predict_btn = st.button("🚀 Predict Pitch Angle", use_container_width=True)

    with col_output:
        st.subheader("Prediction Result")

        if predict_btn:
            try:
                pitch, raw = call_predict_api(wind_speed, rotor_speed, power)

                if pitch is None:
                    st.error("Backend did not return a 'pitch' field. Check API response format.")
                else:
                    st.metric(
                        label="Predicted Pitch Angle",
                        value=f"{pitch:.2f} °",
                        help="Output from the SmartPitch MLP model.",
                    )

                    st.markdown("### Details")
                    st.write(
                        f"- **Wind speed:** {wind_speed:.2f} m/s  \n"
                        f"- **Rotor speed:** {rotor_speed:.2f} RPM  \n"
                        f"- **Power:** {power:.0f} kW"
                    )

                    # 🔍 Comparison with true value (if using a real sample)
                    true_pitch = st.session_state.get("true_pitch")
                    if true_pitch is not None:
                        st.markdown("### 🔬 Comparison with Real Data")
                        st.write(f"- **True pitch (dataset):** `{true_pitch:.2f} °`")
                        st.write(f"- **Model prediction:** `{pitch:.2f} °`")
                        st.write(f"- **Absolute error:** `{abs(pitch - true_pitch):.2f} °`")

                    # 🔹 Save scenario to history
                    st.session_state["history"].append(
                        {
                            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Wind [m/s]": round(wind_speed, 2),
                            "Rotor [RPM]": round(rotor_speed, 2),
                            "Power [kW]": round(power, 1),
                            "Pitch [°]": round(pitch, 2),
                        }
                    )

                    with st.expander("🔍 Raw API response"):
                        st.json(raw)

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Cannot connect to the API.\n\n"
                    "Make sure FastAPI is running:\n"
                    "`uvicorn api.main:app --reload`"
                )
            except Exception as e:
                st.error(f"Unexpected error while calling API: {e}")
        else:
            st.info("Set the input values on the left and click **Predict Pitch Angle** to see the result.")

    st.markdown("---")
    st.subheader("📚 Scenario History")

    history = st.session_state.get("history", [])

    if history:
        df_history = pd.DataFrame(history)
        st.dataframe(df_history, use_container_width=True)

        col_h1, _ = st.columns([1, 3])
        with col_h1:
            if st.button("🧹 Clear history"):
                st.session_state["history"].clear()
                st.rerun()
    else:
        st.caption("No scenarios yet. Run a prediction to start building the history.")

# ========= TAB 2: PROJECT OVERVIEW =========
with tab_overview:
    st.subheader("🎯 Objective")
    st.markdown(
        """
The objective of this project is to design a **data-driven pitch controller** for a utility-scale wind
turbine operating in **Region 3** (above rated wind speed).  
Instead of using only classical lookup tables or PID controllers, we train a supervised ML model
to approximate the mapping:

> (`wind_speed`, `rotor_speed`, `power`) → `pitch`.
"""
    )

    st.subheader("📊 Dataset & Preprocessing")
    st.markdown(
        """
- Source: SCADA data from a simulated or real 5 MW onshore turbine.  
- Region 3 filtering:  
  - `wind_speed ≥ 10 m/s`  
  - `rotor_speed ≥ 9 RPM`  
- Features used:
  - `wind_speed` [m/s]  
  - `rotor_speed` [RPM]  
  - `power` [kW]  
- Target: `pitch` [°]  
- Scaling: `StandardScaler` on features only.  
"""
    )

    st.subheader("🧠 Models")
    with st.expander("Click to see compared models"):
        st.markdown(
            """
We compared multiple regression models:

- Linear Regression  
- K-Nearest Neighbors (KNN)  
- Random Forest Regressor  
- Gradient Boosting Regressor  
- Support Vector Regressor (SVR)  
- Multilayer Perceptron (MLPRegressor) – baseline + tuned  

Best trade-off between accuracy and speed was obtained with the **MLPRegressor**,  
which we then exported and deployed in this API.
"""
        )

    st.subheader("⚙️ Deployment Architecture")
    st.markdown(
        """
1. **Training notebooks** (data prep, training, evaluation).  
2. **Saved artifacts**:  
   - `mlp_pitch_regressor.joblib`  
   - `standard_scaler.joblib`  
3. **FastAPI backend**:  
   - `/health`, `/model_info`, `/predict`.  
4. **Streamlit UI (this app)**:
   - Calls the backend and visualizes predictions for different operating points.
"""
    )

# ========= TAB 3: REGION 3 BASICS =========
with tab_region3:
    st.subheader("🌬️ What is Region 3?")
    st.markdown(
        """
A horizontal-axis wind turbine is typically divided into 3 main operating regions:

1. **Region 1** – below cut-in wind speed, turbine is idle.  
2. **Region 2** – below rated wind speed, controller maximizes power capture (MPPT).  
3. **Region 3** – **above rated wind speed**, controller keeps power near rated by increasing pitch.

In **Region 3**, the main control objective is:

> *Maintain rated electrical power and protect the turbine from excessive loads*  
> by **pitching the blades** (changing the angle of attack).
"""
    )

    st.subheader("🎚️ Why pitch control in Region 3?")
    st.markdown(
        """
- As wind speed increases above rated, the aerodynamic torque would become too large.  
- To compensate, the controller **increases pitch angle**, which reduces lift and limits torque.  
- This keeps:
  - Generator power close to its rated limit.  
  - Structural loads within allowable limits.  
"""
    )

    st.subheader("🤖 Why use Machine Learning?")
    with st.expander("Click to see motivation"):
        st.markdown(
            """
Traditional controllers rely on:
- Fixed **lookup tables** of pitch vs wind speed, or  
- Gain-scheduled PID controllers.

These approaches can struggle when:
- Turbulence intensity changes.  
- The turbine is derated.  
- The turbine ages or operating conditions drift.

By learning directly from **SCADA data**, SmartPitch can:
- Capture nonlinear relationships between wind, rotor speed, power, and pitch.  
- Adapt if retrained on new data.  
- Provide fast inference suitable for real-time assistance or as a decision-support tool.
"""
        )
