# train_delay_model_v2.py
from pathlib import Path
import json, math, re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib

_HAS_CATBOOST = False
_HAS_LGBM = False
try:
    from catboost import CatBoostRegressor, Pool
    _HAS_CATBOOST = True
except Exception:
    pass
try:
    import lightgbm as lgb
    _HAS_LGBM = True
except Exception:
    pass

BASE = Path("trafiklab_sim")
GTFS = BASE / "gtfs_static"
ART  = BASE / "model_artifacts_v2"
ART.mkdir(parents=True, exist_ok=True)

_TIME_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})")

def to_seconds_any(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x)
    ts = pd.to_datetime(s, errors="coerce")
    if ts is not None and not pd.isna(ts):
        return int(ts.hour)*3600 + int(ts.minute)*60 + int(ts.second
    )
    m = _TIME_RE.search(s)
    if m:
        h, mnt, sec = map(int, m.groups())
        return h*3600 + mnt*60 + sec
    return np.nan

def is_rush_hour(h: int) -> int:
    return 1 if (7 <= h <= 9) or (15 <= h <= 18) else 0

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1,lon1,lat2,lon2])
    dlat = lat2-lat1; dlon = lon2-lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

def target_encode(series: pd.Series, y: np.ndarray, k: int = 5, noise: float = 0.02):
    idx = np.arange(len(series))
    folds = np.mod(idx, k)
    global_mean = float(np.mean(y))
    enc = np.zeros_like(y, dtype=float)
    for f in range(k):
        tr = folds != f; te = folds == f
        means = pd.DataFrame({"cat": series[tr].values, "y": y[tr]}).groupby("cat")["y"].mean()
        enc[te] = series[te].map(means).fillna(global_mean).values
    if np.std(enc) > 0:
        enc += np.random.normal(0, noise*np.std(enc), size=len(enc))
    return enc, global_mean

def load_data():
    ml = pd.read_csv(BASE / "ml_stop_events.csv")
    trips = pd.read_csv(GTFS / "trips.txt", dtype=str)
    st = pd.read_csv(GTFS / "stop_times.txt", dtype=str)
    routes = pd.read_csv(GTFS / "routes.txt", dtype=str)
    stops = pd.read_csv(GTFS / "stops.txt", dtype=str)
    st["stop_sequence"] = st["stop_sequence"].astype(int)

    ml = (ml
          .merge(trips[["trip_id","route_id","direction_id"]], on="trip_id", how="left")
          .merge(st[["trip_id","stop_id","stop_sequence","arrival_time","departure_time"]],
                 on=["trip_id","stop_id"], how="left")
          .sort_values(["trip_id","stop_sequence"]).reset_index(drop=True))

    ts = pd.to_datetime(ml["timestamp"], errors="coerce")
    ml["hour"] = ts.dt.hour.fillna(12).astype(int)
    ml["dow"] = ts.dt.dayofweek.fillna(2).astype(int)
    ml["is_weekend"] = (ml["dow"] >= 5).astype(int)
    ml["is_rush"] = ml["hour"].apply(is_rush_hour).astype(int)
    ml["ts_dt"] = ts
    return ml, trips, st, routes, stops

def build_edge_hop_minutes(trips, st):
    st2 = st.copy()
    st2["arr_s"] = st2["arrival_time"].apply(to_seconds_any)
    st2["dep_s"] = st2["departure_time"].apply(to_seconds_any)
    st2 = st2.merge(trips[["trip_id","route_id","direction_id"]], on="trip_id", how="left")
    st2 = st2.sort_values(["trip_id","stop_sequence"])
    st2["arr_s_next"] = st2.groupby("trip_id")["arr_s"].shift(-1)
    st2["stop_id_next"] = st2.groupby("trip_id")["stop_id"].shift(-1)
    st2["hop_min"] = (st2["arr_s_next"] - st2["dep_s"]) / 60.0
    st2 = st2.dropna(subset=["stop_id_next"])
    edge_hops = (st2.groupby(["route_id","direction_id","stop_id","stop_id_next"], as_index=False)
                    .agg(hop_min_med=("hop_min","median"),
                         hop_min_mean=("hop_min","mean")))
    edge_hops["hop_min_med"] = edge_hops["hop_min_med"].clip(0.5, 60)
    edge_hops["hop_min_mean"] = edge_hops["hop_min_mean"].clip(0.5, 60)
    return edge_hops

def build_segments(ml, st, stops, edge_hops):
    st_times = st.copy()
    st_times["arr_s"] = st_times["arrival_time"].apply(to_seconds_any)
    st_times["dep_s"] = st_times["departure_time"].apply(to_seconds_any)
    st_times = st_times.sort_values(["trip_id","stop_sequence"])

    segs = []
    stop_coords = stops.set_index("stop_id")[["stop_lat","stop_lon"]].astype(float).to_dict(orient="index")

    for tid, g in ml.groupby("trip_id", sort=False):
        g = g.sort_values("stop_sequence").reset_index(drop=True)
        if len(g) < 2: continue
        inc = g["arrival_delay_sec"].values[1:] - g["departure_delay_sec"].values[:-1]

        origin_dep_delay = g["departure_delay_sec"].values[:-1]
        hours = g["hour"].values[:-1]
        rush = g["is_rush"].values[:-1]
        dow = g["dow"].values[:-1]
        weekend = g["is_weekend"].values[:-1]
        states = g["state"].astype(str).values[:-1]
        temp = g["temp_c"].astype(float).values[:-1]
        wind = g["wind_ms"].astype(float).values[:-1]
        vis  = g["visibility_km"].astype(float).values[:-1]
        rain = g["is_rain"].astype(int).values[:-1]
        snow = g["is_snow"].astype(int).values[:-1]
        route = g["route_id"].astype(str).values[:-1]
        direc = g["direction_id"].astype(str).fillna("0").values[:-1]
        o_ids = g["stop_id"].astype(str).values[:-1]
        d_ids = g["stop_id"].astype(str).values[1:]
        o_seq = g["stop_sequence"].values[:-1]
        d_seq = g["stop_sequence"].values[1:]
        t_origin = g["ts_dt"].values[:-1]

        lag1 = np.r_[0.0, inc[:-1]]

        this_st = st_times[st_times["trip_id"] == tid]
        dep_map = dict(zip(this_st["stop_sequence"], this_st["dep_s"]))
        arr_map = dict(zip(this_st["stop_sequence"], this_st["arr_s"]))
        hop_min = []
        for s in o_seq:
            dep_s = dep_map.get(s); arr_next = arr_map.get(s+1)
            if dep_s is not None and arr_next is not None and not (np.isnan(dep_s) or np.isnan(arr_next)):
                hop = max(0, arr_next - dep_s) / 60.0
            else:
                hop = np.nan
            hop_min.append(hop)
        hop_min = np.array(hop_min, dtype=float)

        hop_df = edge_hops.set_index(["route_id","direction_id","stop_id","stop_id_next"])
        hop_fallback = []
        for r,dn,o,d in zip(route, direc, o_ids, d_ids):
            key = (r,dn,o,d)
            if key in hop_df.index:
                hop_fallback.append(float(hop_df.loc[key,"hop_min_med"]))
            else:
                hop_fallback.append(3.5)
        hop_min = np.where(np.isnan(hop_min), np.array(hop_fallback), hop_min)
        hop_min = np.clip(hop_min, 0.5, 60.0)

        df_tmp = pd.DataFrame({"route_id":route, "origin_stop_id":o_ids, "t":t_origin})
        headway = np.zeros(len(df_tmp), dtype=float)
        for (rid, sid), grp in df_tmp.groupby(["route_id","origin_stop_id"]):
            idx = grp.sort_values("t").index
            diffs = pd.Series(grp.loc[idx,"t"]).diff().dt.total_seconds().fillna(600)/60.0
            headway[idx] = np.clip(diffs.values, 1.0, 120.0)

        dist_km = []
        for o,d in zip(o_ids,d_ids):
            co = stop_coords.get(o, {"stop_lat":np.nan,"stop_lon":np.nan})
            cd = stop_coords.get(d, {"stop_lat":np.nan,"stop_lon":np.nan})
            if np.isnan(co["stop_lat"]) or np.isnan(cd["stop_lat"]):
                dist_km.append(0.5)
            else:
                dist_km.append(haversine_km(co["stop_lat"],co["stop_lon"],cd["stop_lat"],cd["stop_lon"]))
        dist_km = np.array(dist_km, dtype=float)

        delay_rate = inc / np.maximum(hop_min*60.0, 30.0) * 60.0
        delay_rate = np.clip(delay_rate, -60.0, 300.0)

        segs.append(pd.DataFrame({
            "trip_id": tid,
            "route_id": route,
            "direction_id": direc,
            "origin_stop_id": o_ids,
            "dest_stop_id": d_ids,
            "origin_seq": o_seq,
            "dest_seq": d_seq,
            "hour": hours,
            "is_rush": rush,
            "dow": dow,
            "is_weekend": weekend,
            "state": states,
            "temp_c": temp,
            "wind_ms": wind,
            "visibility_km": vis,
            "is_rain": rain,
            "is_snow": snow,
            "sched_hop_min": hop_min,
            "distance_km": dist_km,
            "headway_min": headway,
            "lag1_inc_sec": lag1,
            "origin_dep_delay_sec": origin_dep_delay,
            "inc_delay_sec": inc,
            "target_rate_sec_per_min": delay_rate
        }))

    seg = pd.concat(segs, ignore_index=True)

    seg = seg.sort_values(
        ["route_id","direction_id","origin_stop_id","dest_stop_id","hour","origin_seq"]
    ).reset_index(drop=True)

    grp = seg.groupby(["route_id","direction_id","origin_stop_id","dest_stop_id"], sort=False)
    def _exp_mean_shifted(s: pd.Series) -> pd.Series:
        return s.shift().expanding(min_periods=5).mean()
    hist = grp["target_rate_sec_per_min"].apply(_exp_mean_shifted)
    hist = hist.reset_index(level=[0,1,2,3], drop=True).reindex(seg.index)
    seg["edge_hist_rate_mean"] = hist.fillna(seg["target_rate_sec_per_min"].median())

    seg["headway_min"] = seg["headway_min"].clip(1, 120)
    seg["origin_dep_delay_sec"] = seg["origin_dep_delay_sec"].clip(-300, 1800)
    seg["lag1_inc_sec"] = seg["lag1_inc_sec"].clip(-300, 1800)
    return seg

def train_and_evaluate(seg):
    y = seg["target_rate_sec_per_min"].values
    X_num = seg[[
        "hour","is_rush","dow","is_weekend",
        "temp_c","wind_ms","visibility_km","is_rain","is_snow",
        "sched_hop_min","distance_km","headway_min",
        "lag1_inc_sec","origin_dep_delay_sec","edge_hist_rate_mean"
    ]].copy()
    cats = seg[["route_id","direction_id","origin_stop_id","dest_stop_id","state"]].astype(str)
    metrics = {"mae": [], "r2": []}

    if _HAS_CATBOOST:
        X_full = pd.concat([cats, X_num], axis=1)
        cat_cols = ["route_id","direction_id","origin_stop_id","dest_stop_id","state"]
        groups = seg["trip_id"].astype(str)
        gkf = GroupKFold(n_splits=5)
        for tr, va in gkf.split(X_full, y, groups):
            tr_pool = Pool(X_full.iloc[tr], y[tr], cat_features=[X_full.columns.get_loc(c) for c in cat_cols])
            va_pool = Pool(X_full.iloc[va], y[va], cat_features=[X_full.columns.get_loc(c) for c in cat_cols])
            m = CatBoostRegressor(loss_function="MAE", depth=8, learning_rate=0.08, iterations=1200,
                                  l2_leaf_reg=3.0, random_seed=42, verbose=False)
            m.fit(tr_pool, eval_set=va_pool)
            p = m.predict(va_pool)
            metrics["mae"].append(float(mean_absolute_error(y[va], p)))
            metrics["r2"].append(float(r2_score(y[va], p)))
        final_pool = Pool(X_full, y, cat_features=[X_full.columns.get_loc(c) for c in cat_cols])
        final_model = CatBoostRegressor(loss_function="MAE", depth=8, learning_rate=0.08, iterations=1200,
                                        l2_leaf_reg=3.0, random_seed=42, verbose=False)
        final_model.fit(final_pool)
        model_type = "catboost"
        schema = {"columns": list(X_full.columns), "cat_cols": cat_cols}

    elif _HAS_LGBM:
        X_full = pd.concat([cats, X_num], axis=1)
        cat_cols = ["route_id","direction_id","origin_stop_id","dest_stop_id","state"]
        cat_idx = [X_full.columns.get_loc(c) for c in cat_cols]
        groups = seg["trip_id"].astype(str)
        gkf = GroupKFold(n_splits=5)
        for tr, va in gkf.split(X_full, y, groups):
            m = lgb.LGBMRegressor(objective="mae", num_leaves=63, learning_rate=0.08,
                                  n_estimators=1200, subsample=0.9, colsample_bytree=0.9, random_state=42)
            m.fit(X_full.iloc[tr], y[tr],
                  eval_set=[(X_full.iloc[va], y[va])], eval_metric="l1",
                  categorical_feature=cat_idx, verbose=False)
            p = m.predict(X_full.iloc[va])
            metrics["mae"].append(float(mean_absolute_error(y[va], p)))
            metrics["r2"].append(float(r2_score(y[va], p)))
        final_model = lgb.LGBMRegressor(objective="mae", num_leaves=63, learning_rate=0.08,
                                        n_estimators=1200, subsample=0.9, colsample_bytree=0.9, random_state=42)
        final_model.fit(X_full, y, categorical_feature=cat_idx)
        model_type = "lightgbm"
        schema = {"columns": list(X_full.columns), "cat_cols": cat_cols, "cat_idx": cat_idx}

    else:
        X_te = X_num.copy()
        enc_meta = {}
        for col in ["route_id","direction_id","origin_stop_id","dest_stop_id","state"]:
            enc, gmean = target_encode(cats[col], y, k=5)
            X_te[f"enc_{col}"] = enc
            enc_meta[col] = {"global_mean": float(gmean)}
        groups = seg["trip_id"].astype(str)
        gkf = GroupKFold(n_splits=5)
        for tr, va in gkf.split(X_te, y, groups):
            m = HistGradientBoostingRegressor(loss="absolute_error", max_depth=10,
                                              learning_rate=0.08, max_iter=600, random_state=42)
            m.fit(X_te.iloc[tr], y[tr])
            p = m.predict(X_te.iloc[va])
            metrics["mae"].append(float(mean_absolute_error(y[va], p)))
            metrics["r2"].append(float(r2_score(y[va], p)))
        final_model = HistGradientBoostingRegressor(loss="absolute_error", max_depth=10,
                                                    learning_rate=0.08, max_iter=600, random_state=42)
        final_model.fit(X_te, y)
        model_type = "sk_hgbr_te"
        schema = {
            "columns": list(X_te.columns),
            "enc_meta": enc_meta,
            "enc_maps": {
                c: seg.groupby(c)["target_rate_sec_per_min"].mean().astype(float).to_dict()
                for c in ["route_id","direction_id","origin_stop_id","dest_stop_id","state"]
            }
        }

    report = {
        "model_type": model_type,
        "cv_mae_rate_sec_per_min_mean": float(np.mean(metrics["mae"])),
        "cv_mae_rate_sec_per_min_std":  float(np.std(metrics["mae"])),
        "cv_r2_mean": float(np.mean(metrics["r2"])),
        "cv_r2_std":  float(np.std(metrics["r2"])),
        "n_segments": int(len(seg))
    }
    joblib.dump(final_model, ART / "segment_delay_rate_model.pkl")
    joblib.dump(schema, ART / "schema.pkl")
    (ART / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("Artifacts saved to:", str(ART.resolve()))

def build_route_orders(trips, st):
    st_full = st.merge(trips[["trip_id","route_id","direction_id"]], on="trip_id", how="left")
    orders = {}
    for (rid, did), g in st_full.groupby(["route_id","direction_id"]):
        trip_pick = g.groupby("trip_id")["stop_sequence"].max().idxmax()
        ordered = g[g["trip_id"]==trip_pick].sort_values("stop_sequence")["stop_id"].tolist()
        orders.setdefault(rid, {})[str(did)] = ordered
    return orders

def main():
    ml, trips, st, routes, stops = load_data()
    edge_hops = build_edge_hop_minutes(trips, st)
    seg = build_segments(ml, st, stops, edge_hops)

    # We'll simply call train_and_evaluate:
    train_and_evaluate(seg)

    route_orders = build_route_orders(trips, st)
    (ART / "route_stop_order.json").write_text(json.dumps(route_orders, indent=2))
    edge_map = (edge_hops
                .assign(direction_id=edge_hops["direction_id"].astype(str))
                .rename(columns={"stop_id":"origin_stop_id","stop_id_next":"dest_stop_id"}))
    edge_map.to_csv(ART / "edge_hop_avgs.csv", index=False)

    print("Artifacts saved to:", str(ART.resolve()))

if __name__ == "__main__":
    main()
