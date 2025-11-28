import os
import datetime
import requests

import numpy as np          # 👈 NEW
import pandas as pd
import streamlit as st

# 🔗 Backend API URL (change if needed)
API_URL = os.getenv("API_URL", "http://localhost:8000")


# Physical pitch limits (must match backend)
PITCH_MIN = 0.0
PITCH_MAX = 30.0
PITCH_EPS = 0.05  # tolerance to detect saturation

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
    payload = {
        "wind_speed": wind_speed,
        "rotor_speed": rotor_speed,
        "power": power,
    }
    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
    resp.raise_for_status()
    data = resp.json()
    pitch = data.get("pitch")           # clipped (safe) command
    pitch_raw = data.get("pitch_raw")   # unconstrained model output (may be >30°)
    return pitch, pitch_raw, data


def call_wind_profile_api(hor_windv: float, rot_speed: float, gen_pwr: float):
    """
    Call the /predict_wind_profile endpoint (Random Forest wind-profile model).

    Backend (Swagger) expects JSON of the form:
    {
      "wind_speeds": [0],
      "rotor_speed": 0,
      "gen_pwr": 0,
      "time_step": 1
    }
    """
    payload = {
        "wind_speeds": [hor_windv],   # list, even for one value
        "rotor_speed": rot_speed,
        "gen_pwr": gen_pwr,
        "time_step": 1,
    }
    resp = requests.post(f"{API_URL}/predict_wind_profile", json=payload, timeout=5)
    resp.raise_for_status()
    data = resp.json()
    pitch = data.get("pitch")
    return pitch, data


# 🔹 NEW: helper for time-series wind-profile
def call_wind_profile_series_api(
    wind_speeds: list[float],
    rotor_speeds: list[float],
    gen_powers: list[float],
    time_step: float = 1.0,
):
    """
    Call the /predict_wind_profile_series endpoint.

    Expects all three lists to have the same length.
    """
    payload = {
        "wind_speeds": wind_speeds,
        "rotor_speeds": rotor_speeds,
        "gen_powers": gen_powers,
        "time_step": time_step,
    }
    resp = requests.post(
        f"{API_URL}/predict_wind_profile_series", json=payload, timeout=10
    )
    resp.raise_for_status()
    data = resp.json()
    pitches = data.get("pitch_series", [])
    return pitches, data


def _parse_series_text(text: str) -> list[float]:
    """
    Small utility: parse comma- or semicolon-separated numbers into a float list.
    Example: '15, 15.5, 16; 17' -> [15.0, 15.5, 16.0, 17.0]
    """
    if not text.strip():
        return []
    parts = text.replace(";", ",").split(",")
    values = []
    for p in parts:
        s = p.strip()
        if not s:
            continue
        values.append(float(s))
    return values


@st.cache_data
def load_region3_data():
    """Load the cleaned Region 3 dataset for sampling."""
    return pd.read_csv("data/region3_clean.csv")


def run_what_if_sweep(
    rotor_speed: float,
    power: float,
    ws_min: float = 10.0,
    ws_max: float = 25.0,
    ws_step: float = 0.5,
):
    """
    Sweep wind speed and call /predict_batch once.
    Returns DataFrame with [wind_speed, rotor_speed, power, pitch, pitch_raw].
    """
    wind_speeds = np.arange(ws_min, ws_max + 1e-9, ws_step)

    samples = [
        {
            "wind_speed": float(ws),
            "rotor_speed": float(rotor_speed),
            "power": float(power),
        }
        for ws in wind_speeds
    ]

    payload = {"samples": samples}

    resp = requests.post(f"{API_URL}/predict_batch", json=payload, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()

    results = data.get("results", [])

    records = []
    for ws, res in zip(wind_speeds, results):
        records.append(
            {
                "wind_speed": ws,
                "rotor_speed": rotor_speed,
                "power": power,
                "pitch": res.get("pitch"),
                "pitch_raw": res.get("pitch_raw"),
            }
        )

    return pd.DataFrame(records)


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
    st.caption("Region 3 collective pitch control using supervised ML (MLPRegressor)")

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
        st.image(
            turbine_path,
            use_container_width=True,
            caption="Onshore Wind Turbines",
        )
    except Exception:
        st.markdown(
            "*(Turbine illustration – add `assets/turbine_region3.jpg` for an image here.)*"
        )
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

tab_console, tab_overview, tab_region3, tab_what_if, tab_wind_profile, tab_wind_profile_series = st.tabs(
    [
        "🎛 SmartPitch Console",
        "📘 Project Overview",
        "🌬️ Region 3 Basics",
        "🧪 What-if Lab",
        "🌪 Wind-profile Model",
        "📈 Wind-profile Time Series",   # NEW TAB
    ]
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
        prev_p = default_p

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
                pitch, pitch_raw, raw = call_predict_api(
                    wind_speed, rotor_speed, power
                )
                if pitch is None:
                    st.error(
                        "Backend did not return a 'pitch' field. Check API response format."
                    )
                else:
                    st.metric(
                        label="Predicted Pitch Angle",
                        value=f"{pitch:.2f} °",
                        help="Output from the SmartPitch MLP model.",
                    )

                # Warn if prediction is saturated at physical limits
                if pitch is not None:
                    if pitch >= PITCH_MAX - PITCH_EPS:
                        if pitch_raw is not None:
                            st.warning(
                                f"Raw model pitch = {pitch_raw:.2f}°. For safety, the commanded "
                                f"pitch is saturated at {PITCH_MAX:.0f}° within the physical "
                                "limits [0°, 30°]."
                            )
                        else:
                            st.warning(
                                f"Pitch angle has been saturated at the physical maximum of {PITCH_MAX:.0f}°. "
                                "This means the operating point is extreme and the unconstrained model "
                                "would suggest a higher pitch, but we limit it for safety."
                            )
                    elif pitch <= PITCH_MIN + PITCH_EPS:
                        if pitch_raw is not None:
                            st.warning(
                                f"Raw model pitch = {pitch_raw:.2f}°. For safety, the commanded "
                                f"pitch is saturated at {PITCH_MIN:.0f}° within the physical "
                                "limits [0°, 30°]."
                            )
                        else:
                            st.warning(
                                f"Pitch angle has been saturated at the physical minimum of {PITCH_MIN:.0f}°. "
                                "This indicates an extreme operating point and the unconstrained model "
                                "would suggest a lower pitch, but we limit it for safety."
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
                        st.write(
                            f"- **Absolute error:** `{abs(pitch - true_pitch):.2f} °`"
                        )

                    # 🔹 Save scenario to history
                    st.session_state["history"].append(
                        {
                            "Timestamp": datetime.datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
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
            st.info(
                "Set the input values on the left and click **Predict Pitch Angle** to see the result."
            )

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


# ========= TAB 4: WHAT-IF LAB =========
with tab_what_if:
    st.subheader("🧪 What-if Lab")

    st.markdown(
        """
Explore how the **predicted pitch** changes when you sweep wind speed in Region 3,
while keeping **rotor speed** and **active power** fixed.
"""
    )

    col1, col2 = st.columns(2)
    with col1:
        fixed_rotor = st.number_input(
            "Fixed rotor speed [RPM]",
            min_value=0.0,
            max_value=25.0,
            value=float(st.session_state.get("rotor_speed", 12.0)),
            step=0.1,
        )
    with col2:
        fixed_power = st.number_input(
            "Fixed active power [kW]",
            min_value=0.0,
            max_value=5000.0,
            value=float(st.session_state.get("power", 4000.0)),
            step=50.0,
        )

    with st.expander("Wind speed sweep settings", expanded=False):
        ws_min = st.number_input("Wind speed min [m/s]", 0.0, 40.0, 10.0, 0.5)
        ws_max = st.number_input("Wind speed max [m/s]", 0.0, 40.0, 25.0, 0.5)
        ws_step = st.number_input("Wind speed step [m/s]", 0.1, 10.0, 0.5, 0.1)

    # 🔁 Run sweep only when button is pressed, but SAVE result in session_state
    if st.button("Run What-if sweep"):
        if ws_max <= ws_min:
            st.error("Wind speed max must be greater than min.")
        else:
            with st.spinner("Sweeping wind speed and calling /predict..."):
                df_sweep = run_what_if_sweep(
                    rotor_speed=fixed_rotor,
                    power=fixed_power,
                    ws_min=ws_min,
                    ws_max=ws_max,
                    ws_step=ws_step,
                )
            st.session_state["what_if_df"] = df_sweep

    # 📦 Use last sweep result if available
    df_sweep = st.session_state.get("what_if_df")

    if df_sweep is not None and not df_sweep.empty:
        st.subheader("Results table")
        st.dataframe(df_sweep, use_container_width=True)

        st.subheader("Pitch vs Wind Speed")

        pitch_choice = st.radio(
            "Pitch to plot:",
            ["Clipped pitch (physical command)", "Raw pitch (model output)"],
            index=0,
            horizontal=True,
        )

        y_col = "pitch" if pitch_choice.startswith("Clipped") else "pitch_raw"

        df_plot = df_sweep.dropna(subset=[y_col])

        if df_plot.empty:
            st.warning(
                f"No valid values found for `{y_col}`. "
                "Check that the backend returns both `pitch` and `pitch_raw`."
            )
        else:
            chart_data = df_plot[["wind_speed", y_col]].set_index("wind_speed")
            st.line_chart(chart_data)

        csv = df_sweep.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv,
            file_name="what_if_pitch_vs_wind.csv",
            mime="text/csv",
        )
    else:
        st.info("Run the sweep to see the curve.")


# ========= TAB 5: SINGLE-POINT WIND-PROFILE MODEL =========
with tab_wind_profile:
    st.subheader("🌪 Wind-profile ML model (Region 3)")

    st.markdown(
        """
This tab uses the **Random Forest wind-profile model** deployed at
`/predict_wind_profile`.  
Enter one operating point (wind speed, rotor speed, generator power) and see the
predicted pitch angle.
"""
    )

    col_in, col_out = st.columns([1, 1])

    with col_in:
        hor_windv = st.number_input(
            "Horizontal wind speed [m/s]",
            min_value=10.0,
            max_value=25.0,
            value=float(st.session_state.get("wind_speed", 15.0)),
            step=0.1,
        )
        rot_speed = st.number_input(
            "Rotor speed [RPM]",
            min_value=9.0,
            max_value=20.0,
            value=float(st.session_state.get("rotor_speed", 12.0)),
            step=0.1,
        )
        gen_pwr = st.number_input(
            "Active power [kW]",
            min_value=0.0,
            max_value=6000.0,
            value=float(st.session_state.get("power", 4000.0)),
            step=50.0,
        )

        wind_profile_btn = st.button("🚀 Predict with wind-profile model")

    with col_out:
        st.subheader("Prediction Result")

        if wind_profile_btn:
            try:
                pitch_wp, raw_wp = call_wind_profile_api(hor_windv, rot_speed, gen_pwr)
                if pitch_wp is None:
                    st.error("Backend did not return a 'pitch' field. Check API.")
                else:
                    st.metric(
                        label="Predicted pitch (wind-profile RF model)",
                        value=f"{pitch_wp:.2f} °",
                    )

                    st.markdown("### Details")
                    st.write(
                        f"- **Horizontal wind speed:** {hor_windv:.2f} m/s  \n"
                        f"- **Rotor speed:** {rot_speed:.2f} RPM  \n"
                        f"- **Generator power:** {gen_pwr:.0f} kW"
                    )

                    with st.expander("🔍 Raw API response"):
                        st.json(raw_wp)

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Cannot connect to the API.\n\n"
                    "Make sure FastAPI is running:\n"
                    "`uvicorn api.main:app --reload`"
                )
            except Exception as e:
                st.error(f"Unexpected error while calling wind-profile API: {e}")
        else:
            st.info(
                "Set the inputs on the left and click "
                "**Predict with wind-profile model**."
            )


# ========= TAB 6: TIME-SERIES WIND-PROFILE MODEL =========
with tab_wind_profile_series:
    st.subheader("📈 Wind-profile Time Series (Region 3)")

    st.markdown(
        """
Provide a **time series** of wind speed, rotor speed and generator power.
All three lists must have the same number of samples.

The app will call `/predict_wind_profile_series` and plot the predicted pitch profile.
"""
    )

    example_ws = "15, 15.5, 16, 16.5, 17"
    example_rs = "12, 12.1, 12.2, 12.3, 12.4"
    example_gp = "3800, 3900, 4000, 4100, 4200"

    col_ts1, col_ts2 = st.columns(2)

    with col_ts1:
        ws_text = st.text_area(
            "Wind speeds [m/s] (comma or ; separated)",
            value=example_ws,
            height=100,
        )
        rs_text = st.text_area(
            "Rotor speeds [RPM] (same length)",
            value=example_rs,
            height=100,
        )
        gp_text = st.text_area(
            "Generator powers [kW] (same length)",
            value=example_gp,
            height=100,
        )

    with col_ts2:
        time_step = st.number_input(
            "Time step between samples [s]",
            min_value=0.1,
            max_value=600.0,
            value=1.0,
            step=0.1,
        )

        run_ts_btn = st.button("🚀 Predict time-series pitch profile")

    if run_ts_btn:
        try:
            ws_list = _parse_series_text(ws_text)
            rs_list = _parse_series_text(rs_text)
            gp_list = _parse_series_text(gp_text)

            n_ws = len(ws_list)
            n_rs = len(rs_list)
            n_gp = len(gp_list)

            if n_ws == 0:
                st.error("Please provide at least one sample in the wind-speed series.")
            elif not (n_ws == n_rs == n_gp):
                st.error(
                    f"All series must have the SAME length. "
                    f"Now: wind={n_ws}, rotor={n_rs}, power={n_gp}"
                )
            else:
                with st.spinner("Calling /predict_wind_profile_series ..."):
                    pitch_series, raw_series = call_wind_profile_series_api(
                        ws_list, rs_list, gp_list, time_step=time_step
                    )

                if not pitch_series:
                    st.error("Backend did not return 'pitch_series'. Check API.")
                else:
                    n = len(ws_list)
                    times = np.arange(0, n * time_step, time_step)

                    df_ts = pd.DataFrame(
                        {
                            "time_s": times[:n],
                            "wind_speed": ws_list,
                            "rotor_speed": rs_list,
                            "gen_power": gp_list,
                            "pitch": pitch_series,
                        }
                    )

                    st.subheader("Time-series table")
                    st.dataframe(df_ts, use_container_width=True)

                    st.subheader("Pitch vs Time")
                    chart_data = df_ts[["time_s", "pitch"]].set_index("time_s")
                    st.line_chart(chart_data)

                    with st.expander("🔍 Raw API response"):
                        st.json(raw_series)

        except ValueError as ve:
            st.error(f"Could not parse the series: {ve}")
        except requests.exceptions.ConnectionError:
            st.error(
                "❌ Cannot connect to the API.\n\n"
                "Make sure FastAPI is running:\n"
                "`uvicorn api.main:app --reload`"
            )
        except Exception as e:
            st.error(f"Unexpected error while calling time-series API: {e}")
