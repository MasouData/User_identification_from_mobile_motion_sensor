# Data_Science_Assignment

ThreatFabric Data Science Challenge — **User identification from mobile motion sensor sessions**.

This repository contains:
- **Code** (Databricks / PySpark + Python) to build session-level features and train models
- **submission.csv** with predicted `user_id` for each `session_id` in `test.csv`
- **Images** used in the short report (EDA + data structure)

---

## Project Objective

Given raw event-level motion sensor data (accelerometer, gyroscope, rotation, etc.), build a system that identifies the most likely user behind each session in the test set (closed-set user identification). 

---

## Data Overview

Each row is a **sensor event** with:
- `timestamp` (Unix ms)
- `sensor_type` (1,2,4,5,6,19)
- `field_0 ... field_7` (sensor-specific values; many are null depending on sensor)
- `session_id`
- `user_id` (train only)

Sensor reference and field meanings are provided in the assignment description. 

---

## Approach Summary

### 1) Data Understanding & EDA
- Verified **one user per session** (no mixed-user sessions)
- Checked missing values: expected due to sensor-specific schemas
- Computed session size percentiles (events per session)
- Verified class balance by sessions: **15 sessions per user** (imbalance ratio = 1.0)

**Figures**
- `images/session_structure.png` — relationship: user → sessions → sensor events
- `images/sessions_per_user.png` — sessions per user (balanced)

### 2) Feature Engineering (Session-level)
Converted event-level streams into a **single feature row per session**:

For each `(session_id, sensor_type)`:
- `n_events`
- `duration_ms = max(timestamp) - min(timestamp)`
- `event_rate = n_events / (duration_ms + 1)`
- Per-field statistics for `field_0..field_7`: mean/std/min/max + non-null count (`*_nn`)
- Physics-informed magnitude (for sensors with x/y/z axes):
  - `mag_012 = sqrt(field_0^2 + field_1^2 + field_2^2)`
  - aggregated with mean/std/min/max + count

Then pivoted `sensor_type` into wide columns so each session is one row.

### 3) Modeling & Evaluation
Metric: **Macro-F1**  
Validation: **Repeated stratified splits (5 seeds)**

Models compared:
- Dummy baselines (most_frequent, stratified)
- Logistic Regression
- Linear SVM
- Random Forest
- **XGBoost (selected final)**

Final model: **XGBoost**, trained on the full training set and used to predict `user_id` for the test sessions.

---

## Repository Structure

- `main.py`  
  Main notebook-export script with:
  - EDA
  - Feature engineering (PySpark)
  - Model comparison
  - XGBoost tuning + final training
  - `submission.csv` generation

- `submission.csv`  
  Final predictions for `test.csv` (required deliverable). 

- `images/`  
  Plots and diagrams used in the report.

<!-- - `REPORT.md` (recommended)  
  Short report with assumptions, key insights, feature strategy, modeling results, and business recommendations. -->

---

## How to Run (Databricks)

1. Upload `train.csv` and `test.csv` to a Databricks Volume.
2. Update paths in `main.py`:
   - `/Volumes/.../train.csv`
   - `/Volumes/.../test.csv`
3. Run the notebook/script end-to-end.
4. Confirm `submission.csv` is created and push to GitHub.

---

## Business Notes

Behavioral motion signatures can support:
- passive authentication
- user consistency monitoring across sessions
- anomaly / fraud detection when a session deviates from a user baseline 
