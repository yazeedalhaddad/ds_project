
# Trafiklab Trip Delay Model v2 (Bundle)

This bundle contains the training, inference, and testing scripts for a route-aware bus delay predictor.
It uses simulated/collected stop-level events (`ml_stop_events.csv`) and GTFS static files.

## Folder layout
```
trafiklab_sim/
  ├─ train_delay_model_v2.py
  ├─ predict_delay_v2.py
  ├─ build_edge_rate_baseline.py
  ├─ test_model_v2.py
  ├─ model_artifacts_v2/         # outputs saved here after training
  └─ gtfs_static/                # put your GTFS CSVs here
      ├─ trips.txt
      ├─ stop_times.txt
      ├─ stops.txt
      └─ routes.txt
```
**You must provide:**
- `trafiklab_sim/ml_stop_events.csv`
- `trafiklab_sim/gtfs_static/trips.txt`
- `trafiklab_sim/gtfs_static/stop_times.txt`
- `trafiklab_sim/gtfs_static/stops.txt`
- `trafiklab_sim/gtfs_static/routes.txt`

### Quick start

1) Place your CSV files in the paths above.
2) (Optional but recommended) install CatBoost and LightGBM:
   ```bash
   pip install catboost lightgbm
   ```
3) Train the v2 model:
   ```bash
   python3 trafiklab_sim/train_delay_model_v2.py
   ```
   Artifacts will be written to `trafiklab_sim/model_artifacts_v2/`.
4) Build the per-edge rate baseline for inference blending:
   ```bash
   python3 trafiklab_sim/build_edge_rate_baseline.py
   ```
5) Run real tests (smoke + backtest):
   ```bash
   python3 trafiklab_sim/test_model_v2.py
   ```

### Using the predictor in your app
```python
from trafiklab_sim.predict_delay_v2 import DelayPredictorV2

pred = DelayPredictorV2("trafiklab_sim/model_artifacts_v2")
wx = {"state":"rain","temp_c":6,"wind_ms":7.5,"visibility_km":5.2,"is_rain":1,"is_snow":0, "dow":4}
delay_sec = pred.predict_between_stops(
    route_id="T", origin_stop_id="S001", dest_stop_id="S010",
    departure_hour=8, weather=wx,
    origin_departure_delay_sec=30, headway_min=12, dow=4, is_weekend=0
)
print("Predicted delay (s):", round(delay_sec,1))
```

### CSV schemas (columns required)
- **ml_stop_events.csv** (at minimum):  
  `trip_id, stop_id, timestamp, arrival_delay_sec, departure_delay_sec, state, temp_c, wind_ms, visibility_km, is_rain, is_snow`
- **gtfs_static/stop_times.txt**:  
  `trip_id, stop_id, stop_sequence, arrival_time, departure_time` (times can be HH:MM:SS or ISO)
- **gtfs_static/trips.txt**:  
  `trip_id, route_id, direction_id`
- **gtfs_static/stops.txt**:  
  `stop_id, stop_name, stop_lat, stop_lon`
- **gtfs_static/routes.txt**:  
  `route_id, route_short_name, route_long_name`

### Notes
- The v2 model predicts **delay rate (sec/min)** per hop and sums to total seconds.
- Inference blends the **per-edge historical prior** with the model’s prediction using evidence-weighting.
- GroupKFold by trip avoids leakage. You can switch to time-based CV if you have service-day IDs.
