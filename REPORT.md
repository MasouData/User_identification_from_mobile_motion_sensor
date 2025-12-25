# ThreatFabric Data Science Assignment — Short Report

## 1. Assumptions & Thought Process

- **Closed-set identification:** the goal is to predict a `user_id` for each test `session_id`, assuming test sessions belong to the same set of users observed in training (multiclass classification).  
- **Session consistency:** each `session_id` corresponds to exactly one user in the training set; this was validated by checking that no session maps to multiple users.
- **Sensor schema:** missing values in `field_0..field_7` are expected because each sensor type exposes a different subset of fields (e.g., rotation sensor has more fields than accelerometer).  
- **Robustness to timing:** timestamps are used to derive duration and event-rate features, without assuming a fixed sampling frequency.

## 2. Key Insights from EDA

- The dataset contains motion events from **6 sensor types** (1, 2, 4, 5, 6, 19) consistent with the provided sensor documentation.
- **Balanced classes at the session level:** each user contributes **15 sessions** (imbalance ratio = 1.0). This reduces the risk of a majority-class dominance in training.
- Session lengths (events per session) vary substantially (p50/p90/p99 shown in the notebook), motivating normalization features such as `event_rate`.

Figures used:
- `images/session_structure.png` — illustrates the data hierarchy (user → sessions → sensor events).
- `images/sessions_per_user.png` — confirms balanced sessions per user.

## 3. Feature Engineering Strategy

### Goal
Transform event-level sensor streams into a **single row per session** suitable for standard ML models.

### Aggregation & Pivot
For each `(session_id, sensor_type)`, compute:
- `n_events`
- `duration_ms = max(timestamp) - min(timestamp)`
- `event_rate = n_events / (duration_ms + 1)`
- For each `field_0..field_7`: `mean`, `std`, `min`, `max`, and non-null count (`*_nn`)  
  (`*_nn` acts as a “presence” signal since some sensors do not populate all fields.)

Then pivot `sensor_type` to wide format so each session becomes one feature vector.

### Physics-informed features (low risk)
To better capture motion intensity independent of axis direction, compute:
- `mag_012 = sqrt(field_0^2 + field_1^2 + field_2^2)`
and aggregate `mag_mean`, `mag_std`, `mag_min`, `mag_max`, and `mag_nn` per `(session_id, sensor_type)`.

For `sensor_type=6` (Rotation sensor), “orientation stability” is represented via aggregated statistics of pitch/roll/yaw fields (`field_5/field_6/field_7`) included automatically in the per-field stats.

## 4. Modeling Approach & Performance

### Metric & validation
- Primary metric: **Macro-F1** (appropriate for multiclass evaluation).
- Validation: **Repeated stratified holdout** (5 random seeds, 80/20 split) to reduce single-split variance.

### Baselines & models
Compared:
- Dummy baselines: `most_frequent`, `stratified`
- Logistic Regression (scaled)
- Linear SVM (scaled)
- Random Forest
- XGBoost (multiclass)

### Results (mean macro-F1 over 5 seeds)
- Dummy (most frequent): ~0.005  
- Dummy (stratified): ~0.051  
- Logistic Regression: ~0.838  
- Linear SVM: ~0.819  
- Random Forest: ~0.916  
- **XGBoost (selected): ~0.952**

**Conclusion:** XGBoost achieved the strongest and most consistent performance after adding magnitude + duration/rate features, suggesting user identity is driven by nonlinear combinations of motion statistics across sensors.

## 5. Business Implications & Recommendations

This approach can support “behavioral biometrics” scenarios such as:
- **Passive authentication:** verify whether a session’s motion signature matches the claimed user.
- **Fraud / anomaly detection:** flag sessions with low model confidence or strong deviation from a user’s historical pattern.
- **Risk scoring:** use prediction confidence (e.g., soft probabilities) as an input to a broader fraud decision pipeline.

Recommendations for productionizing:
- Track feature drift (device models, OS updates, sensor calibration changes).
- Retrain periodically and monitor performance per user segment.
- Store derived session features rather than raw sensor streams where possible to reduce privacy exposure and storage costs.

## 6. Limitations & Future Work

- Add more time-aware dynamics (e.g., lagged differences, jerk) using timestamp ordering if needed.
- Explore calibration by device type and sensor availability.
- Consider probability calibration (e.g., Platt scaling / isotonic) if using confidence scores operationally.
