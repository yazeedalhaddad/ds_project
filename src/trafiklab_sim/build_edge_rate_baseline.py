
# build_edge_rate_baseline.py
from pathlib import Path
import numpy as np, pandas as pd, re

BASE = Path("trafiklab_sim")
ART  = BASE / "model_artifacts_v2"
GTFS = BASE / "gtfs_static"
ART.mkdir(parents=True, exist_ok=True)

def to_seconds_any(x):
    ts = pd.to_datetime(x, errors="coerce")
    if ts is not None and not pd.isna(ts):
        return ts.hour*3600 + ts.minute*60 + ts.second
    m = re.search(r"(\d{1,2}):(\d{2}):(\d{2})", str(x))
    if m:
        h, mnt, s = map(int, m.groups()); return h*3600 + mnt*60 + s
    return np.nan

ml = pd.read_csv(BASE/"ml_stop_events.csv")
trips = pd.read_csv(GTFS/"trips.txt", dtype=str)
st = pd.read_csv(GTFS/"stop_times.txt", dtype=str)
st["stop_sequence"] = st["stop_sequence"].astype(int)

df = (ml.merge(trips[["trip_id","route_id","direction_id"]], on="trip_id", how="left")
        .merge(st[["trip_id","stop_id","stop_sequence","arrival_time","departure_time"]],
               on=["trip_id","stop_id"], how="left")
        .sort_values(["trip_id","stop_sequence"]).reset_index(drop=True))

rows = []
for tid, g in df.groupby("trip_id", sort=False):
    g = g.sort_values("stop_sequence")
    if len(g) < 2: continue
    inc = g["arrival_delay_sec"].values[1:] - g["departure_delay_sec"].values[:-1]
    route = g["route_id"].astype(str).values[:-1]
    direc = g["direction_id"].astype(str).fillna("0").values[:-1]
    o_ids = g["stop_id"].astype(str).values[:-1]
    d_ids = g["stop_id"].astype(str).values[1:]
    arr_s = g["arrival_time"].apply(to_seconds_any).values
    dep_s = g["departure_time"].apply(to_seconds_any).values
    hop_min = (arr_s[1:] - dep_s[:-1]) / 60.0
    hop_min = np.where(np.isfinite(hop_min), np.clip(hop_min, 0.5, 60), 3.5)
    rate = inc / np.maximum(hop_min*60.0, 30.0) * 60.0
    rate = np.clip(rate, -60.0, 300.0)
    rows.append(pd.DataFrame({
        "route_id": route, "direction_id": direc,
        "origin_stop_id": o_ids, "dest_stop_id": d_ids,
        "edge_rate_med": rate
    }))

edge_rates = pd.concat(rows, ignore_index=True)
agg = (edge_rates
       .groupby(["route_id","direction_id","origin_stop_id","dest_stop_id"], as_index=False)
       .agg(edge_rate_med=("edge_rate_med","median"),
            edge_rate_mean=("edge_rate_med","mean"),
            n=("edge_rate_med","size")))
agg["direction_id"] = agg["direction_id"].astype(str)
agg.to_csv(ART/"edge_rate_baseline.csv", index=False)

global_med = float(edge_rates["edge_rate_med"].median())
(ART/"edge_rate_global_median.txt").write_text(str(global_med))

print("Saved:", ART/"edge_rate_baseline.csv")
print("Global median (sec/min):", round(global_med,2))
