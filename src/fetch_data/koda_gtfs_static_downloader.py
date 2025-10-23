import os
import time
import requests
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
# The base URL for the KoDa GTFS Static API endpoint.
BASE_URL = "https://api.koda.trafiklab.se/KoDa/api/v2/gtfs-static/sl"

# Directory where the downloaded zip files will be saved.
OUTPUT_DIR = "gtfs_static_data"

# Maximum number of concurrent download tasks. Trafiklab recommends 2-3.
MAX_WORKERS = 2

# Time in seconds to wait between polling checks if a file is being prepared.
POLLING_INTERVAL_SECONDS = 30

# Request timeout in seconds.
REQUEST_TIMEOUT_SECONDS = 60


def generate_date_range(start_str, end_str):
    """Generates a list of date strings in YYYY-MM-DD format for the given range."""
    start_date = date.fromisoformat(start_str)
    end_date = date.fromisoformat(end_str)
    delta = timedelta(days=1)
    dates = []
    current_date = start_date
    while current_date >= end_date:
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date -= delta
    return dates


def download_data_for_day(date_str, api_key):
    """
    Handles the complete download process for a single day.
    This includes polling for file readiness and then downloading the content.
    """
    url = f"{BASE_URL}?date={date_str}&key={api_key}"
    output_filename = f"GTFS-SL-{date_str}.zip"
    output_filepath = os.path.join(OUTPUT_DIR, output_filename)

    # 1. Skip if the file already exists
    if os.path.exists(output_filepath):
        print(f"[{date_str}] SKIPPED: File already exists at {output_filepath}")
        return date_str, "skipped"

    print(f"[{date_str}] INFO: Starting process...")

    # 2. Poll the API until the file is ready for download (status 200)
    try:
        while True:
            # Use a HEAD request for a lightweight status check
            head_response = requests.head(url, timeout=REQUEST_TIMEOUT_SECONDS)

            if head_response.status_code == 200:
                print(f"[{date_str}] INFO: File is ready for download.")
                break
            elif head_response.status_code == 202:
                print(f"[{date_str}] INFO: File is being prepared. Checking again in {POLLING_INTERVAL_SECONDS}s...")
                time.sleep(POLLING_INTERVAL_SECONDS)
            else:
                # Handle other potential errors like 404 Not Found, 401 Unauthorized etc.
                error_msg = f"[{date_str}] FAILED: Received unexpected status code {head_response.status_code} during polling."
                print(error_msg)
                return date_str, error_msg

    except requests.exceptions.RequestException as e:
        error_msg = f"[{date_str}] FAILED: Network error during polling: {e}"
        print(error_msg)
        return date_str, error_msg

    # 3. Download the actual file content
    try:
        print(f"[{date_str}] INFO: Downloading to {output_filepath}...")
        with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT_SECONDS) as r:
            r.raise_for_status()
            with open(output_filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        success_msg = f"[{date_str}] SUCCESS: Successfully downloaded."
        print(success_msg)
        return date_str, "success"

    except requests.exceptions.RequestException as e:
        error_msg = f"[{date_str}] FAILED: Error during file download: {e}"
        print(error_msg)
        # Clean up partially downloaded file if it exists
        if os.path.exists(output_filepath):
            os.remove(output_filepath)
        return date_str, error_msg


def main():
    """Main function to orchestrate the download process."""
    print("--- Trafiklab KoDa GTFS-RT Downloader ---")

    # Securely get the API key from the user
    api_key = input("Please enter your Trafiklab API key: ").strip()
    if not api_key:
        print("API key cannot be empty. Exiting.")
        return

    start_date = "2025-06-30"
    end_date = "2022-06-30"

    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dates_to_download = generate_date_range(start_date, end_date)
    total_files = len(dates_to_download)
    print(f"\nFound {total_files} days to process from {start_date} to {end_date}.")
    print(f"Concurrent download tasks set to: {MAX_WORKERS}\n")

    # Use ThreadPoolExecutor to manage concurrent downloads
    results = {"success": [], "skipped": [], "failed": []}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a future for each download task
        future_to_date = {executor.submit(download_data_for_day, date_str, api_key): date_str for date_str in dates_to_download}

        for i, future in enumerate(as_completed(future_to_date)):
            date_str = future_to_date[future]
            try:
                _, status = future.result()
                if status == "success":
                    results["success"].append(date_str)
                elif status == "skipped":
                    results["skipped"].append(date_str)
                else:
                    results["failed"].append((date_str, status))
            except Exception as exc:
                print(f"[{date_str}] FAILED: An unexpected error occurred: {exc}")
                results["failed"].append((date_str, str(exc)))
            
            # Progress update
            processed_count = i + 1
            print(f"\n--- Progress: {processed_count}/{total_files} tasks completed ---\n")


    # Final summary
    print("\n--- Download Process Finished ---")
    print(f"Successful downloads: {len(results['success'])}")
    print(f"Skipped (already existed): {len(results['skipped'])}")
    print(f"Failed downloads: {len(results['failed'])}")
    if results["failed"]:
        print("\nDetails of failed downloads:")
        for date_str, reason in results["failed"]:
            print(f" - {date_str}: {reason}")
    print("---------------------------------")


if __name__ == "__main__":
    main()

