# Region 3 ML Pitch Controller

This project implements a pure ML-based collective pitch controller for wind turbines operating in Region 3, using SCADA data.

## Project Structure

- `notebooks/`: Jupyter notebooks for data preparation, training, and evaluation.
- `app/backend/`: FastAPI backend exposing the ML model as an API.
- `app/frontend/`: Streamlit UI to interact with the controller.
- `models/`: Saved trained models (e.g., `ml_pitch_controller.joblib`, `feature_scaler.joblib`).
- `data/`: Input datasets (raw and processed).
- `docs/`: Any additional documentation or figures.

## How to Run Backend Locally

1. Create a virtual environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Train your model in the notebooks and save it to `models/ml_pitch_controller.joblib`.
   Save your fitted scaler to `models/feature_scaler.joblib`.

3. Start the API:

   ```bash
   uvicorn app.backend.main:app --reload
   ```

## How to Run Streamlit UI

With the backend running on `http://localhost:8000`:

```bash
streamlit run app/frontend/streamlit_app.py
```

## Docker (Backend)

To build and run the backend in Docker:

```bash
docker build -t region3-pitch-backend .
docker run -p 8000:8000 region3-pitch-backend
```
