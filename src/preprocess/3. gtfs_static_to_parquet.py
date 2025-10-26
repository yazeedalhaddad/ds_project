import zipfile, duckdb, os, glob

zip_dir = "../../data/raw/gtfs_static/"
out_dir = "../../data/interim/gtfs_static_parquet/"

os.makedirs(out_dir, exist_ok=True)

for zip_path in glob.glob(f"{zip_dir}/GTFS-SL-*.zip"):
    basename = os.path.basename(zip_path)
    date_str = basename.replace("GTFS-SL-", "").replace(".zip", "")
    dest = os.path.join(out_dir, date_str)
    os.makedirs(dest, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        for name in ["trips.txt", "routes.txt", "stop_times.txt"]:
            z.extract(name, dest)

    con = duckdb.connect()
    for txt in ["trips", "routes", "stop_times"]:
        csv_file = f"{dest}/{txt}.txt"
        parquet_file = f"{dest}/{txt}.parquet"

        con.execute(f"""
        COPY (
            SELECT * FROM read_csv_auto('{csv_file}', ALL_VARCHAR=True, strict_mode=False)
        ) TO '{parquet_file}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """)

        # Delete the original text file
        os.remove(csv_file)
