# ThreatFabric Data Science Assignment — Short Report

## 1. Assumptions & Thought Process
- **Closed-set identification:** predict a `user_id` for each test `session_id`, assuming test sessions belong to the same set of users seen in training (multiclass classification).
- **Session consistency:** each `session_id` maps to exactly one user in training; validated by checking no session contains multiple `user_id`s.
- **Sensor schema:** missing values in `field_0..field_7` are expected because each sensor type populates a different subset of fields (e.g., rotation has more fields than accelerometer).
- **Timing:** timestamps are used only to derive duration and event-rate features (no fixed sampling-rate assumptions).

## 2. Key Insights from EDA
- Data contains motion events from **6 sensor types** (1, 2, 4, 5, 6, 19) consistent with the provided sensor documentation.
- **Balanced labels at the session level:** each user contributes **15 sessions** (imbalance ratio = 1.0).
- Session lengths (events per session) vary notably (p50/p90/p99 shown in the notebook), motivating features such as `duration_ms` and `event_rate`.
- Missingness is strongly **sensor-dependent**, so non-null counts (`*_nn`) are informative features rather than simply “data quality issues”.

Figures:
- `images/session_structure.png` — data hierarchy (user → sessions → sensor events)
- `images/sessions_per_user.png` — sessions per user (balanced)

## 3. Feature Engineering Strategy

### Goal
Transform event-level streams into a **single feature vector per session** usable by standard ML classifiers.

### Aggregation per (session_id, sensor_type)
For each sensor stream inside a session:
- `n_events`
- `duration_ms = max(timestamp) - min(timestamp)`
- `event_rate = n_events / (duration_ms + 1)`
- For each `field_0..field_7`:  
  `mean`, `median (p50)`, `std`, `min`, `max`, and non-null count (`*_nn`)
  - Median is included for **robustness to sensor spikes/outliers**.
  - `*_nn` acts as a “presence/density” signal because sensors differ in which fields they populate.

### Motion intensity (axis-invariant)
To capture motion strength regardless of device orientation:
- `mag_012 = sqrt(field_0^2 + field_1^2 + field_2^2)`
Aggregated per sensor stream as: `mag_mean`, `mag_p50`, `mag_std`, `mag_min`, `mag_max`, `mag_nn`.

### Pivot to session-level table
Sensor aggregates are pivoted so each session becomes one row with columns like:
- `sensor_2_field_0_mean`, `sensor_6_field_7_std`, `sensor_19_mag_p50`, etc.

This produces a fixed-width feature table for modeling.

## 4. Modeling Approach & Performance

### Metric
Primary metric: **Macro-F1**, to weight all users equally and penalize models that perform poorly on a subset of users.

### Validation
- **Repeated stratified holdout** using identical splits across models (same indices per seed).
- 20 seeds (80/20 split) to reduce variance from a single random split.
- **Paired Wilcoxon signed-rank test** used to compare top models (RF vs XGB) on the paired seed results.

### Models Compared
- Dummy baselines: `most_frequent`, `stratified`
- Logistic Regression (scaled)
- Linear SVM (scaled)
- Random Forest
- XGBoost (multiclass)

### Summary of results (macro-F1)
- Dummy baselines are near zero (expected).
- Tree-based models outperform linear baselines due to nonlinear interactions across sensor statistics.
- **Random Forest was selected as final** after repeated evaluation: it achieved the highest mean Macro-F1 and significantly outperformed XGBoost under paired testing (Wilcoxon p < 0.01).

## 5. Business Implications & Recommendations
This approach supports behavioral biometrics use cases such as:
- **Passive authentication:** verify whether a session’s motion signature matches the claimed user.
- **Fraud / anomaly detection:** flag sessions that deviate from a user baseline or produce low model confidence.
- **Risk scoring:** use model confidence (e.g., class probabilities) as an input into a broader decision pipeline.

Recommendations for productionization:
- Monitor feature drift (device models, OS updates, sensor calibration differences).
- Retrain periodically and track performance over time.
- Store derived session features rather than raw streams where possible to reduce storage and privacy exposure.

## 6. Limitations & Future Work
- Add richer time-aware features (jerk, autocorrelation, spectral features) using timestamp ordering if needed.
- Incorporate device metadata if available (phone model, OS) to control for sensor differences.
- Calibrate probabilities if confidence will be used operationally (Platt / isotonic calibration).