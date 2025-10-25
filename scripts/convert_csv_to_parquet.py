import polars as pl

# Streaming CSV read
scan = pl.scan_csv("data/raw/weather/weather.csv")

# Collect in streaming mode and write to Parquet
scan.sink_parquet("data/interim/weather/weather.parquet")
