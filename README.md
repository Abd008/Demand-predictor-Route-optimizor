# Smart Bus Demand Prediction & Route Optimization

An end-to-end bus operations prototype that combines passenger-counting computer vision, machine-learning demand prediction, and route-level bus allocation recommendations.

The system is designed around two cooperating processes:

1. A **FastAPI backend** predicts passenger demand and exposes route status.
2. A **camera or Streamlit frontend** reports occupancy and visualizes predictions.

## Problem and Project Goal

Public transport operators often have limited visibility into how many passengers are waiting at each stop and how crowded buses will become along a route. This can lead to overcrowded buses during peak periods, insufficient service on busy routes, and underused buses when demand is low. These issues make journeys less reliable for passengers and make bus allocation harder for operators.

This project tries to solve that problem by building a prototype that:

- counts passengers from a live camera feed,
- combines camera occupancy with ticketing and route information,
- predicts passenger demand at upcoming stops,
- estimates crowd levels and next-hour route demand, and
- recommends whether the current bus allocation should be increased, reduced, or maintained.

The intended outcome is an operational decision-support tool: instead of reacting only after a bus becomes overcrowded, an operator can use predicted demand and live occupancy to plan bus deployment earlier. The project is a proof of concept rather than a complete production transport-management system; routes are currently configured in code and historical data storage, authentication, and multi-camera tracking would be required for a larger deployment.

## Features

- Random Forest demand prediction using stop, date, time, day, boarding, and deboarding data.
- Contextual prediction adjustments for:
  - nearby events,
  - recent ticket activity,
  - live camera occupancy.
- Real-time passenger detection from a webcam using the best available detector:
  - YOLOv8, when `ultralytics` is installed,
  - OpenCV DNN SSD,
  - HOG/SVM fallback.
- Rolling detection smoothing and non-maximum suppression to reduce duplicate or noisy counts.
- Route status for configured routes, including:
  - current stop,
  - occupancy,
  - next-hour demand,
  - crowd level,
  - current and required buses.
- Streamlit pages for route simulation and camera monitoring.
- A lightweight sample backend for demonstrations when the trained model is unavailable.

## Architecture

```text
Webcam / camera process ──POST /predict-demand──> FastAPI backend
                                                        │
                                                        ├─ demand model
                                                        ├─ contextual adjustments
                                                        └─ /routes-status
                                                               │
                                                               ▼
                                                   Streamlit dashboard
```

## Requirements

- Python 3.9 or later
- A webcam for camera features
- Dependencies in [`requirements.txt`](requirements.txt)
- Optional: `ultralytics` for improved YOLOv8 person detection

## Installation

```bash
git clone https://github.com/Abd008/Demand-predictor-Route-optimizor.git
cd Demand-predictor-Route-optimizor

python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

For the optional YOLO detector:

```bash
pip install ultralytics
```

## Quick Start

### 1. Start the API

From the repository root:

```bash
python -m uvicorn backend.api.app:app --reload --host 0.0.0.0 --port 8000
```

Useful URLs:

- Health check: <http://127.0.0.1:8000/health>
- Interactive API documentation: <http://127.0.0.1:8000/docs>
- Route status: <http://127.0.0.1:8000/routes-status>

### 2. Run the Streamlit dashboard

Open another terminal with the virtual environment activated:

```bash
streamlit run frontend/dashboard.py
```

The dashboard polls `/routes-status` and displays route summaries, stop-level demand, crowd levels, and bus recommendations. Additional pages are available for route simulation and camera monitoring.

### 3. Run the standalone camera client

```bash
python bus_camera.py
```

Common options:

```bash
python bus_camera.py --station "Majestic Bus Stand"
python bus_camera.py --camera 1
python bus_camera.py --capacity 60
python bus_camera.py --conf 0.30
python bus_camera.py --api http://192.168.1.100:8000/predict-demand
```

Press **Q** in the camera window to exit.

The legacy OpenCV monitor can also be started directly:

```bash
python opencv_bus_monitor.py
```

## API

### `GET /health`

Returns a simple liveness response:

```json
{"status": "backend running"}
```

### `POST /predict-demand`

Required fields:

```json
{
  "station_name": "Yeshwanthapura T.T.M.C.",
  "boarding": 5,
  "deboarding": 1,
  "first_ticket_time": "08:15",
  "day": "Monday",
  "date": "2026-11-21"
}
```

Optional real-time fields:

```json
{
  "ticket_boarding_count": 12,
  "occupancy_percentage": 82.0,
  "events_data": {
    "has_event": true,
    "event_popularity": "high",
    "distance_km": 1.5
  }
}
```

Example response:

```json
{
  "stop_name": "Yeshwanthapura T.T.M.C.",
  "base_prediction": 6.81,
  "final_prediction": 7,
  "adjustments": {
    "event": 0.0,
    "realtime": 0.0,
    "opencv": 0.0
  }
}
```

### `GET /routes-status`

Returns a live snapshot for the configured routes, including stop predictions, crowd levels, occupancy, predicted next-hour demand, and required bus counts.

## Sample backend

To run a model-free demo backend on port `8001`:

```bash
python backend_sample.py
```

Point the camera client at it:

```bash
python bus_camera.py --api http://127.0.0.1:8001/predict-demand
```

The sample backend is intended for local demos and does not replace the ML backend for production use.

## Project structure

```text
backend/
  api/app.py                 FastAPI application and route status endpoint
  predictor/predictor.py     Feature engineering and demand prediction
  predictor/optimizer.py     Bus allocation recommendations
frontend/
  dashboard.py               Streamlit route dashboard
  pages/1_simulation.py      Route demand simulation
  pages/2_bus_camera.py      Streamlit camera monitor
bus_camera.py                Standalone camera client
opencv_bus_monitor.py        Alternate OpenCV monitor
backend_sample.py            Lightweight rule-based API
model/bus_model.pkl          Serialized prediction model
model/*.csv                 Test/training data sample
deploy.prototxt              OpenCV DNN network definition
```

## Model and fallback behavior

The backend loads `model/bus_model.pkl` once at startup. If the model cannot be loaded, the predictor uses its built-in fallback behavior so the API can still be exercised during development. For reliable predictions, keep the bundled model available and use station names supported by the route data.

Camera detection is similarly fault-tolerant: YOLOv8 is preferred when installed, with OpenCV and HOG-based fallbacks used when it is not.

## Production notes

- Restrict the backend CORS policy instead of allowing all origins.
- Put the API behind HTTPS and a reverse proxy when it is publicly reachable.
- Run the backend with a production server such as Gunicorn with Uvicorn workers.
- Store historical readings in a database if long-term analytics are required.
- Configure routes and station data outside the source code for multi-route deployments.

More detailed operational guidance is available in [`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md), and the model/vision design is described in [`MENTOR_EXPLANATION.md`](MENTOR_EXPLANATION.md).
