# 🛰️ AI-Driven Space Debris Detection and Collision Prediction System

**Using YOLOv8 and Orbital Mechanics for Indian Satellites**

> A BSERC internship project combining computer vision and orbital
> mechanics into a single decision-support dashboard for satellite
> collision-risk awareness, built around real ISRO spacecraft.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ffcc)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🌌 Overview

Space debris is one of the most pressing operational threats to active
satellites. With ISRO expanding its constellation of Earth-observation,
communication, and navigation satellites, automated tools for **detecting**
potential debris objects and **predicting close-approach (conjunction)
risk** are increasingly valuable.

This project delivers an end-to-end demo system with two integrated
pipelines:

| Pipeline | Description |
|---|---|
| 🧠 **Vision** | A YOLOv8 model detects candidate debris objects in telescope/radar-style imagery |
| 🌍 **Orbital Mechanics** | Live TLE data is propagated with SGP4 (via Skyfield) to compute closest-approach distance, relative velocity, and a risk score between a real Indian satellite and a tracked debris/rocket-body object |

Both pipelines are unified in a single **dark-themed Streamlit dashboard**.

---

## ✨ Key Features

- 🛰️ Real Indian satellites: Cartosat-3, RISAT-2B, EOS-04, GSAT-30,
  INSAT-3DR, Oceansat-2, Chandrayaan-2 Orbiter, and more
- 📡 Live TLE fetching from Celestrak with automatic offline fallback
  (never breaks mid-demo, even without internet)
- 🎯 YOLOv8 object detection with bounding boxes + confidence scores;
  auto-generates a synthetic star-field frame if no image is uploaded
- 📊 Closest-approach analysis: miss distance, relative speed, risk
  classification (HIGH / MEDIUM / LOW / NEGLIGIBLE), and an
  educational pseudo-probability score
- 📋 **Batch risk screening** — check one Indian satellite against the
  *entire* debris catalog in one pass, ranked by risk, with CSV export
- 🚨 **Alert stub** — one-click Telegram/email notification when a
  conjunction is classified HIGH risk (safe no-op "demo mode" until you
  add real credentials — see `alerts.py`)
- 🎓 **Custom model training script** (`train_yolo.py`) — fine-tune
  YOLOv8n on a real space-debris imagery dataset instead of relying on
  the generic COCO fallback
- 🌍 Interactive 3D orbit visualization (Plotly) around a reference Earth
- 🌌 Custom dark "mission control" UI theme
- 🛡️ Defensive error handling throughout — network failures, missing
  models, and bad inputs degrade gracefully instead of crashing

---

## 📁 Project Structure

```
space-debris-detection/
├── app.py                     # Streamlit dashboard (entry point)
├── detect.py                  # YOLOv8 debris detection module
├── orbit.py                   # Skyfield/SGP4 orbital mechanics + risk scoring + batch screening
├── alerts.py                  # Telegram/email HIGH-risk alert stub
├── train_yolo.py              # Fine-tune YOLOv8 on a custom debris dataset
├── requirements.txt
├── README.md
├── data/
│   ├── satellites.json        # Real Indian satellite catalog (NORAD IDs)
│   ├── debris_catalog.csv     # Sample tracked debris/rocket bodies
│   └── sample_images/         # Optional sample imagery for the vision demo
├── models/
│   └── yolov8_debris.pt       # (optional) your custom-trained weights
├── utils/
│   └── tle_fetch.py           # Bulk TLE fetch/cache CLI helper
├── assets/
│   └── style.css              # Dark space theme
└── notebooks/
    └── colab_demo.ipynb       # One-click Google Colab demo
```

---

## ⚙️ Tech Stack

- **Frontend/Dashboard:** Streamlit, Plotly
- **Computer Vision:** Ultralytics YOLOv8, OpenCV
- **Orbital Mechanics:** Skyfield (SGP4 propagator), Astropy
- **Data:** Celestrak live TLE API, curated Indian satellite/debris catalogs
- **Language:** Python 3.10+

---

## 🚀 Getting Started (Local)

### 1. Clone and set up environment
```bash
git clone https://github.com/<your-username>/space-debris-detection.git
cd space-debris-detection
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. (Optional) Add a custom YOLOv8 debris model
Place your trained weights at `models/yolov8_debris.pt`. If omitted, the
app automatically falls back to a stock COCO-pretrained YOLOv8n model
for demo purposes and clearly labels detections as illustrative.

### 3. Run the dashboard
```bash
streamlit run app.py
```
Open the printed local URL (typically `http://localhost:8501`).

---

## ☁️ Running on Google Colab

1. Open `notebooks/colab_demo.ipynb` in Colab.
2. Run the cells in order — they clone the repo, install dependencies,
   launch Streamlit in the background, and expose it via `localtunnel`.
3. Click the generated public URL and enter the tunnel password (your
   Colab runtime's public IP, printed in the notebook) when prompted.

This is ideal for judges/evaluators who want to try the live dashboard
without any local setup.

---

## 🎓 Training a Custom YOLOv8 Debris Model (Recommended)

The app runs out of the box with a generic COCO-pretrained fallback, but
a **custom-trained model is the single highest-impact upgrade** for
credibility. Steps:

```bash
pip install kaggle
# Place your Kaggle API token at ~/.kaggle/kaggle.json first
kaggle datasets download -d <owner>/space-debris-detection-dataset-for-yolov8 \
    -p data/raw --unzip

python train_yolo.py --data data/raw/data.yaml --epochs 50 --imgsz 640
```

> ⚠️ Verify the exact dataset slug on Kaggle before downloading — search
> "space debris yolov8" on kaggle.com/datasets, since slugs/owners can
> change over time.

`train_yolo.py` automatically copies the best checkpoint to
`models/yolov8_debris.pt` when training finishes. Restart the app and
`detect.py` will pick it up automatically (the "using fallback model"
warning banner disappears once your custom weights are detected).

---

## 🚨 Setting Up Real Alerts (Optional)

`alerts.py` ships in safe **demo mode** — clicking "Send Alert" logs
what *would* be sent without needing any credentials. To wire up real
notifications, set these as environment variables (or in
`.streamlit/secrets.toml`):

```toml
TELEGRAM_BOT_TOKEN = "123456:ABC-your-bot-token"
TELEGRAM_CHAT_ID   = "987654321"

ALERT_EMAIL_FROM     = "your-alert-bot@gmail.com"
ALERT_EMAIL_TO       = "you@example.com"
ALERT_EMAIL_PASSWORD = "app-specific-password"
```



---


