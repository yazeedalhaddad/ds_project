
# predict_delay_v2.py
from pathlib import Path
import json, joblib
import pandas as pd
import numpy as np

class DelayPredictorV2:
    def __init__(self, artifacts_dir="trafiklab_sim/model_artifacts_v2"):
        self.base = Path(artifacts_dir)
        self.model = joblib.load(self.base / "segment_delay_rate_model.pkl")
        self.schema = joblib.load(self.base / "schema.pkl")
        self.route_order = json.loads((self.base / "route_stop_order.json").read_text())
        self.edge_hops = pd.read_csv(self.base / "edge_hop_avgs.csv")
        self.edge_hops["direction_id"] = self.edge_hops["direction_id"].astype(str)
        er_path = self.base / "edge_rate_baseline.csv"
        if er_path.exists():
            self.edge_rate = pd.read_csv(er_path)
            self.edge_rate["direction_id"] = self.edge_rate["direction_id"].astype(str)
        else:
            self.edge_rate = pd.DataFrame(columns=["route_id","direction_id","origin_stop_id","dest_stop_id","edge_rate_med","n"])
        gmed_path = self.base / "edge_rate_global_median.txt"
        self.global_rate_med = float(gmed_path.read_text()) if gmed_path.exists() else 4.4
        self.columns = self.schema.get("columns", [])
        self.cat_cols = self.schema.get("cat_cols", [])
        self.cat_idx  = self.schema.get("cat_idx", None)
        self.enc_meta = self.schema.get("enc_meta", None)
        self.enc_maps = self.schema.get("enc_maps", None)
        self._family = "sk"
        try:
            from catboost import CatBoostRegressor  # noqa
            if self.cat_cols:
                self._family = "cat"
        except Exception:
            pass
        if self._family != "cat":
            try:
                import lightgbm as lgb  # noqa
                if self.cat_cols:
                    self._family = "lgbm"
            except Exception:
                pass

    def _edge_hop_minutes(self, route_id, direction_id, origin_stop, dest_stop):
        m = self.edge_hops
        sel = m[(m["route_id"]==route_id) & (m["direction_id"]==str(direction_id)) &
                (m["origin_stop_id"]==origin_stop) & (m["dest_stop_id"]==dest_stop)]
        if not sel.empty:
            return float(sel["hop_min_med"].iloc[0])
        return 3.5

    def _edge_rate_prior(self, route_id, direction_id, origin_stop, dest_stop):
        m = self.edge_rate
        if not m.empty:
            sel = m[(m["route_id"]==route_id) & (m["direction_id"]==str(direction_id)) &
                    (m["origin_stop_id"]==origin_stop) & (m["dest_stop_id"]==dest_stop)]
            if not sel.empty:
                prior = float(sel["edge_rate_med"].iloc[0])
                n = int(sel["n"].iloc[0]) if "n" in sel.columns else 0
                return prior, n
        return float(self.global_rate_med), 0

    def _encode_fallback(self, col, val):
        m = self.enc_maps[col]; g = self.enc_meta[col]["global_mean"]
        return float(m.get(str(val), g))

    def _build_row(self, route_id, direction_id, origin_stop, dest_stop, hour, wx,
                   sched_hop_min, lag1_inc_sec, origin_dep_delay_sec, edge_hist_rate_mean,
                   distance_km=0.8, headway_min=10.0, dow=None, is_weekend=None):
        base = {
            "hour": int(hour),
            "is_rush": int(wx.get("is_rush", 1 if (7 <= hour <= 9) or (15 <= hour <= 18) else 0)),
            "dow": int(dow if dow is not None else wx.get("dow", 2)),
            "is_weekend": int(is_weekend if is_weekend is not None else (1 if int(wx.get("dow", 2)) >= 5 else 0)),
            "temp_c": float(wx["temp_c"]),
            "wind_ms": float(wx["wind_ms"]),
            "visibility_km": float(wx["visibility_km"]),
            "is_rain": int(wx.get("is_rain", int(wx["state"]=="rain"))),
            "is_snow": int(wx.get("is_snow", int(wx["state"]=="snow"))),
            "sched_hop_min": float(sched_hop_min),
            "distance_km": float(distance_km),
            "headway_min": float(headway_min),
            "lag1_inc_sec": float(lag1_inc_sec),
            "origin_dep_delay_sec": float(origin_dep_delay_sec),
            "edge_hist_rate_mean": float(edge_hist_rate_mean),
        }
        if self._family in ("cat","lgbm"):
            row = {
                "route_id": str(route_id),
                "direction_id": str(direction_id),
                "origin_stop_id": str(origin_stop),
                "dest_stop_id": str(dest_stop),
                "state": str(wx["state"]),
                **base
            }
            X = pd.DataFrame([row], columns=self.columns)
        else:
            row = {
                **base,
                "enc_route_id": self._encode_fallback("route_id", route_id),
                "enc_direction_id": self._encode_fallback("direction_id", direction_id),
                "enc_origin_stop_id": self._encode_fallback("origin_stop_id", origin_stop),
                "enc_dest_stop_id": self._encode_fallback("dest_stop_id", dest_stop),
                "enc_state": self._encode_fallback("state", wx["state"]),
            }
            X = pd.DataFrame([row], columns=self.columns)
        return X

    def predict_between_stops(self, route_id, origin_stop_id, dest_stop_id, departure_hour, weather,
                              origin_departure_delay_sec=0.0, headway_min=10.0, dow=None, is_weekend=None):
        rid = str(route_id)
        if rid not in self.route_order:
            raise ValueError("Unknown route_id")

        chosen_dir, order = None, None
        for did, seq in self.route_order[rid].items():
            if origin_stop_id in seq and dest_stop_id in seq:
                i0, i1 = seq.index(origin_stop_id), seq.index(dest_stop_id)
                if i1 > i0:
                    chosen_dir, order = did, seq
                    break
        if order is None:
            raise ValueError("Origin must precede destination on one of the route directions.")

        total_sec = 0.0
        lag1_inc_sec = 0.0
        hour = int(departure_hour)
        i0 = order.index(origin_stop_id); i1 = order.index(dest_stop_id)

        for i in range(i0, i1):
            o, d = order[i], order[i+1]
            hop_min = self._edge_hop_minutes(rid, chosen_dir, o, d)
            edge_prior, n_obs = self._edge_rate_prior(rid, chosen_dir, o, d)

            if i == i0 and origin_departure_delay_sec:
                lag1_seed = 0.25 * float(origin_departure_delay_sec)
                lag1_inc_sec = lag1_seed

            X = self._build_row(
                route_id=rid, direction_id=chosen_dir,
                origin_stop=o, dest_stop=d,
                hour=hour, wx=weather,
                sched_hop_min=hop_min,
                lag1_inc_sec=lag1_inc_sec,
                origin_dep_delay_sec=origin_departure_delay_sec if i == i0 else 0.0,
                edge_hist_rate_mean=edge_prior,
                headway_min=headway_min,
                dow=dow, is_weekend=is_weekend
            )

            model_rate = float(self.model.predict(X)[0])   # sec/min
            k = 20.0
            alpha = n_obs / (n_obs + k) if (n_obs is not None and n_obs >= 0) else 0.0
            rate = alpha * edge_prior + (1.0 - alpha) * model_rate

            inc = rate * hop_min
            total_sec += inc
            lag1_inc_sec = inc

        return max(0.0, total_sec)

def debug_onehop(pred: "DelayPredictorV2", route_id, origin_stop, dest_stop, hour, wx,
                 origin_departure_delay_sec=0.0, headway_min=10.0, dow=None, is_weekend=None):
    rid = str(route_id)
    chosen_dir, order = None, None
    for did, seq in pred.route_order[rid].items():
        if origin_stop in seq and dest_stop in seq:
            i0, i1 = seq.index(origin_stop), seq.index(dest_stop)
            if i1 == i0 + 1:
                chosen_dir, order = did, seq
                break
    if order is None:
        raise ValueError("debug_onehop expects adjacent stops.")
    hop_min = pred._edge_hop_minutes(rid, chosen_dir, origin_stop, dest_stop)
    edge_prior, n_obs = pred._edge_rate_prior(rid, chosen_dir, origin_stop, dest_stop)
    X = pred._build_row(
        route_id=rid, direction_id=chosen_dir,
        origin_stop=origin_stop, dest_stop=dest_stop,
        hour=int(hour), wx=wx,
        sched_hop_min=hop_min,
        lag1_inc_sec=0.0,
        origin_dep_delay_sec=float(origin_departure_delay_sec),
        edge_hist_rate_mean=float(edge_prior),
        headway_min=float(headway_min),
        dow=dow, is_weekend=is_weekend
    )
    print("\\n--- DEBUG ONE HOP ---")
    print("Direction:", chosen_dir, "| hop_min:", round(hop_min,2), "min",
          "| edge_prior (sec/min):", round(edge_prior,2), "| n:", n_obs)
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print("Row →")
        print(X.iloc[0])
    model_rate = float(pred.model.predict(X)[0])
    k = 20.0; alpha = n_obs / (n_obs + k) if (n_obs is not None and n_obs >= 0) else 0.0
    blended = alpha * edge_prior + (1.0 - alpha) * model_rate
    inc_sec = blended * hop_min
    print(f"Model rate: {model_rate:.3f} sec/min | alpha={alpha:.2f} | blended: {blended:.3f} sec/min → inc: {inc_sec:.1f} s")
    return inc_sec
