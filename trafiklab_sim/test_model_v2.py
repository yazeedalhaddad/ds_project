
# test_model_v2.py
from pathlib import Path
import random, numpy as np, pandas as pd

ART_DIR = Path("trafiklab_sim/model_artifacts_v2")
BASE    = Path("trafiklab_sim")
GTFS    = BASE / "gtfs_static"

from predict_delay_v2 import DelayPredictorV2, debug_onehop

def load_data():
    ml = pd.read_csv(BASE / "ml_stop_events.csv")
    trips = pd.read_csv(GTFS / "trips.txt", dtype=str)
    st = pd.read_csv(GTFS / "stop_times.txt", dtype=str)
    st["stop_sequence"] = st["stop_sequence"].astype(int)
    return ml, trips, st

def pick_some_pairs(pred, k=3, min_hops=3, max_hops=7):
    cases = []
    for rid, dirs in pred.route_order.items():
        for did, seq in dirs.items():
            if len(seq) < min_hops+1: continue
            i0 = random.randint(0, max(0, len(seq) - (min_hops+1)))
            i1 = min(len(seq)-1, i0 + random.randint(min_hops, min(max_hops, len(seq)-1-i0)))
            origin, dest = seq[i0], seq[i1]
            cases.append((rid, did, origin, dest))
            if len(cases) >= k: return cases
    return cases

def smoke_tests():
    print("\\n=== 1) Smoke tests ===")
    pred = DelayPredictorV2(str(ART_DIR))
    cases = pick_some_pairs(pred, k=4)
    weather_scenarios = [
        {"name":"clear_morning", "wx":{"state":"clear","temp_c":12,"wind_ms":3,"visibility_km":10,"is_rain":0,"is_snow":0, "dow":2}, "hour":8, "dow":2, "weekend":0},
        {"name":"rain_peak",     "wx":{"state":"rain","temp_c":7,"wind_ms":6,"visibility_km":4,"is_rain":1,"is_snow":0, "dow":2},  "hour":8, "dow":2, "weekend":0},
        {"name":"snow_evening",  "wx":{"state":"snow","temp_c":-2,"wind_ms":4,"visibility_km":3,"is_rain":0,"is_snow":1, "dow":4}, "hour":17,"dow":4, "weekend":0},
        {"name":"windy_night",   "wx":{"state":"clouds","temp_c":5,"wind_ms":10,"visibility_km":8,"is_rain":0,"is_snow":0, "dow":6}, "hour":22, "dow":6, "weekend":1},
    ]
    for (rid, did, origin, dest) in cases:
        print(f"\\nRoute {rid} dir {did}: {origin} → {dest}")
        for sc in weather_scenarios:
            delay_sec = pred.predict_between_stops(
                route_id=rid, origin_stop_id=origin, dest_stop_id=dest,
                departure_hour=sc["hour"], weather=sc["wx"],
                origin_departure_delay_sec=30, headway_min=12,
                dow=sc["dow"], is_weekend=sc["weekend"]
            )
            print(f"  [{sc['name']}] hour={sc['hour']:02d}  → predicted {delay_sec/60:.1f} min ({int(delay_sec)} s)")

def build_segment_ground_truth(ml, trips, st):
    df = (ml.merge(trips[["trip_id","route_id","direction_id"]], on="trip_id", how="left")
            .merge(st[["trip_id","stop_id","stop_sequence","arrival_time","departure_time"]],
                   on=["trip_id","stop_id"], how="left")
            .sort_values(["trip_id","stop_sequence"]).reset_index(drop=True))
    segs = []
    for tid, g in df.groupby("trip_id", sort=False):
        g = g.sort_values("stop_sequence").reset_index(drop=True)
        if len(g) < 2: continue
        inc = g["arrival_delay_sec"].values[1:] - g["departure_delay_sec"].values[:-1]
        origin_rows = g.iloc[:-1].copy()
        origin_rows["dest_stop_id"] = g["stop_id"].values[1:]
        origin_rows["inc_delay_sec_true"] = inc
        segs.append(origin_rows)
    seg = pd.concat(segs, ignore_index=True)
    ts = pd.to_datetime(seg["timestamp"], errors="coerce")
    seg["hour"] = ts.dt.hour.fillna(12).astype(int)

    headway = np.zeros(len(seg), dtype=float)
    tmp = pd.DataFrame({"route_id":seg["route_id"].astype(str),
                        "origin_stop_id":seg["stop_id"].astype(str),
                        "ts":ts})
    for (rid, sid), grp in tmp.groupby(["route_id","origin_stop_id"]):
        idx = grp.sort_values("ts").index
        diffs = pd.Series(grp.loc[idx,"ts"]).diff().dt.total_seconds().fillna(600)/60.0
        headway[idx] = np.clip(diffs.values, 1.0, 120.0)
    seg["headway_min"] = headway

    for c in ["state","temp_c","wind_ms","visibility_km","is_rain","is_snow"]:
        if c not in seg.columns:
            if c == "state": seg[c] = "clear"
            elif c in ("is_rain","is_snow"): seg[c] = 0
            elif c == "temp_c": seg[c] = 10.0
            elif c == "wind_ms": seg[c] = 3.0
            elif c == "visibility_km": seg[c] = 8.0

    seg["origin_dep_delay_sec"] = seg["departure_delay_sec"].clip(-300, 1800)
    return seg

def backtest_random_edges(n=200, seed=42):
    print("\\n=== 2) Backtest on random segment edges (origin → next) ===")
    ml, trips, st = load_data()
    seg = build_segment_ground_truth(ml, trips, st)
    if len(seg) == 0:
        print("No edges available to backtest."); return
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(seg.index.values, size=min(n, len(seg)), replace=False)
    sample = seg.loc[sample_idx].copy()
    pred = DelayPredictorV2(str(ART_DIR))
    y_true, y_pred = [], []
    for _, r in sample.iterrows():
        rid = str(r["route_id"]); origin = str(r["stop_id"]); dest = str(r["dest_stop_id"])
        hour = int(r["hour"])
        ts = pd.to_datetime(r["timestamp"], errors="coerce")
        dow = int(ts.dayofweek) if pd.notna(ts) else 2
        is_weekend = int(dow >= 5)
        wx = {"state": str(r.get("state","clear")),
              "temp_c": float(r.get("temp_c",10.0)),
              "wind_ms": float(r.get("wind_ms",3.0)),
              "visibility_km": float(r.get("visibility_km",8.0)),
              "is_rain": int(r.get("is_rain",0)),
              "is_snow": int(r.get("is_snow",0)),
              "dow": dow}
        try:
            p = pred.predict_between_stops(
                route_id=rid, origin_stop_id=origin, dest_stop_id=dest,
                departure_hour=hour, weather=wx,
                origin_departure_delay_sec=float(r["origin_dep_delay_sec"]),
                headway_min=float(r["headway_min"]), dow=dow, is_weekend=is_weekend
            )
        except Exception:
            continue
        y_pred.append(float(p)); y_true.append(float(r["inc_delay_sec_true"]))
    if len(y_true) < 10:
        print(f"Too few comparable edges ({len(y_true)})."); return
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"Backtest edges: {len(y_true)}")
    print(f"MAE (sec): {mae:.2f}")
    print(f"R²: {r2:.3f}")
    print("Examples (first 10):")
    for i in range(min(10, len(y_true))):
        print(f"  pred={y_pred[i]:6.1f}s   true={y_true[i]:6.1f}s   err={y_pred[i]-y_true[i]:6.1f}s")

def edge_cases():
    print("\\n=== 3) Edge cases ===")
    pred = DelayPredictorV2(str(ART_DIR))
    try:
        pred.predict_between_stops("NON_EXISTENT", "S001", "S002", 9,
            {"state":"clear","temp_c":10,"wind_ms":2,"visibility_km":10,"is_rain":0,"is_snow":0})
        print("❌ Expected failure for unknown route, but call succeeded.")
    except Exception as e:
        print("✅ Unknown route properly raises:", str(e))
    rid = next(iter(pred.route_order.keys()))
    did = next(iter(pred.route_order[rid].keys()))
    seq = pred.route_order[rid][did]
    if len(seq) >= 2:
        origin = seq[1]; dest = seq[0]
        try:
            pred.predict_between_stops(rid, origin, dest, 9,
                {"state":"clear","temp_c":10,"wind_ms":2,"visibility_km":10,"is_rain":0,"is_snow":0})
            print("❌ Expected failure for reversed stops, but call succeeded.")
        except Exception as e:
            print("✅ Reversed stops properly raise:", str(e))

if __name__ == "__main__":
    random.seed(7); np.random.seed(7)
    smoke_tests()
    backtest_random_edges(n=300, seed=7)
    edge_cases()
