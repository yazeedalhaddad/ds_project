import duckdb
import os
from tqdm import tqdm

# === Paths ===
GTFS_RT_PARQUET = "../../data/interim/gtfs_rt_parquet/gtfs_rt_snow_filtered.parquet"
GTFS_STATIC_BASE = "../../data/interim/gtfs_static_parquet"
WEATHER_PARQUET = "../../data/interim/weather/weather.parquet"
OUT_DIR = "../../data/interim/joined_daily"

os.makedirs(OUT_DIR, exist_ok=True)

# === Connect DuckDB ===
con = duckdb.connect()

# Load GTFS-RT once
con.execute(f"""
CREATE OR REPLACE TABLE gtfs_rt AS 
SELECT * FROM '{GTFS_RT_PARQUET}';
""")

# Load Weather once
con.execute(f"""
CREATE OR REPLACE TABLE weather AS 
SELECT 
    Date,
    HOD,
    Temperature,
    Precipitation,
    SnowDepth,
    WindSpeed
FROM '{WEATHER_PARQUET}';
""")

# Extract distinct available dates
dates = [d[0] for d in con.sql(
    "SELECT DISTINCT date FROM gtfs_rt ORDER BY date"
).fetchall()]

print(f"🗓 Found {len(dates)} distinct dates in GTFS-RT")

# === Daily Join Loop ===
for d in tqdm(dates, desc="Joining static + RT + weather", unit="day"):
    static_path = f"{GTFS_STATIC_BASE}/{d}"
    out_file = f"{OUT_DIR}/date={d}.parquet"

    if os.path.exists(out_file):
        continue
    if not os.path.exists(static_path):
        print(f"⚠️ Missing static data for {d}, skipping")
        continue

    try:
        query = f"""
        COPY (
          SELECT
            rt.date,
            rt.hod,
            rt.trip_id,
            rt.stop_id,
            rt.delay,
            t.route_id,
            r.route_type,
            t.direction_id,
            st.shape_dist_traveled,
            w.Temperature,
            w.Precipitation,
            w.SnowDepth,
            w.WindSpeed
          FROM gtfs_rt AS rt
          LEFT JOIN '{static_path}/trips.parquet' t USING (trip_id)
          LEFT JOIN '{static_path}/routes.parquet' r USING (route_id)
          LEFT JOIN '{static_path}/stop_times.parquet' st
            ON rt.trip_id = st.trip_id AND rt.stop_id = st.stop_id
          LEFT JOIN weather w
            ON w.Date = rt.date AND w.HOD = rt.hod
          WHERE rt.date = '{d}'
        )
        TO '{out_file}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """
        con.execute(query)

    except Exception as e:
        print(f"❌ Error joining {d}: {e}")
