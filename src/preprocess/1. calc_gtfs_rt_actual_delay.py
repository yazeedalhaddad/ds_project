import os
import re
import py7zr
import pandas as pd
from google.transit import gtfs_realtime_pb2
import tempfile
import shutil

# --- Configuration ---
# Directory containing your compressed sl-TripUpdates-YYYY-MM-DD.7z files.
ARCHIVES_DIR = "gtfs_rt_data"

# The output CSV file name.
OUTPUT_CSV_FILE = "actual_trip_delays_from_archives.csv"


def find_and_sort_pb_files(root_dir):
    """
    Finds all .pb files recursively in a directory and sorts them chronologically.
    """
    pb_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".pb"):
                pb_files.append(os.path.join(root, file))
    pb_files.sort()
    return pb_files


def process_pb_file(filepath):
    """
    Parses a single GTFS-RT protobuf file from a given filepath.
    """
    try:
        with open(filepath, "rb") as f:
            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(f.read())

        file_data = {}
        feed_timestamp = feed.header.timestamp

        for entity in feed.entity:
            if entity.HasField('trip_update'):
                trip_id = entity.trip_update.trip.trip_id
                for stu in entity.trip_update.stop_time_update:
                    key = (trip_id, stu.stop_sequence)
                    arrival = stu.arrival
                    departure = stu.departure
                    file_data[key] = {
                        'trip_id': trip_id,
                        'stop_sequence': stu.stop_sequence,
                        'stop_id': stu.stop_id,
                        'actual_arrival_delay': arrival.delay if arrival.HasField('delay') else None,
                        'arrival_uncertainty': arrival.uncertainty if arrival.HasField('uncertainty') else None,
                        'actual_departure_delay': departure.delay if departure.HasField('delay') else None,
                        'departure_uncertainty': departure.uncertainty if departure.HasField('uncertainty') else None,
                        'timestamp': feed_timestamp,
                    }
        return file_data
    except Exception as e:
        print(f"Warning: Could not parse protobuf file {os.path.basename(filepath)}. Error: {e}")
        return {}


def find_and_sort_7z_files(archives_dir):
    """
    Finds .7z files in the specified directory and sorts them chronologically
    based on the YYYY-MM-DD date in their filenames.
    """
    files_to_process = []
    date_pattern = re.compile(r"sl-TripUpdates-(\d{4}-\d{2}-\d{2})\.7z")

    if not os.path.isdir(archives_dir):
        print(f"ERROR: The specified archives directory does not exist: {archives_dir}")
        return []

    for filename in os.listdir(archives_dir):
        match = date_pattern.match(filename)
        if match:
            files_to_process.append(os.path.join(archives_dir, filename))

    files_to_process.sort()
    print(f"Found {len(files_to_process)} archive files to process.")
    return files_to_process


def main():
    """
    Main function to orchestrate the data processing pipeline from compressed archives.
    """
    print("--- Starting GTFS-RT Data Pipeline from Compressed Archives ---")
    archive_files = find_and_sort_7z_files(ARCHIVES_DIR)

    if not archive_files:
        print("No archives found. Exiting.")
        return

    last_known_state = {}
    total_events_found = 0 # Track total events written to disk
    total_archives = len(archive_files)
    write_header = True # Flag to write header only on the first iteration

    for i, archive_path in enumerate(archive_files):
        print(f"\n--- Processing Archive {i + 1}/{total_archives}: {os.path.basename(archive_path)} ---")
        temp_dir = tempfile.mkdtemp()
        archive_events = [] # Temporary list for events from THIS archive
        try:
            # 1. Extract archive to a temporary directory
            with py7zr.SevenZipFile(archive_path, mode='r') as z:
                print(f"  Extracting to temporary location: {temp_dir}")
                z.extractall(path=temp_dir)

            # 2. Find and sort the extracted protobuf files
            pb_files_in_archive = find_and_sort_pb_files(temp_dir)
            
            if not pb_files_in_archive:
                print("  Warning: No .pb files found in this archive.")
                continue

            total_internal_files = len(pb_files_in_archive)
            print(f"  Found {total_internal_files} protobuf files to process from archive.")

            # 3. Process each file from the temporary directory
            for j, pb_filepath in enumerate(pb_files_in_archive):
                current_data = process_pb_file(pb_filepath)
                current_trip_stops = set(current_data.keys())
                previous_trip_stops = set(last_known_state.keys())
                completed_stops = previous_trip_stops - current_trip_stops
    
                if completed_stops:
                    for key in completed_stops:
                        # Append the completed event to the temporary archive list
                        archive_events.append(last_known_state[key])

                last_known_state = {key: last_known_state[key] for key in last_known_state if key in current_trip_stops}
                last_known_state.update(current_data)
                
                if (j + 1) % 5000 == 0 or (j + 1) == total_internal_files:
                    print(f"  ... Processed {j + 1}/{total_internal_files} files within archive. "
                          f"Events found in this archive so far: {len(archive_events)}")

            # 4. WRITE THE ACCUMULATED ARCHIVE EVENTS TO CSV (Incremental Write)
            if archive_events:
                print(f"  [WRITING] Writing {len(archive_events)} events to disk...")
                
                # Convert buffer to DataFrame
                df = pd.DataFrame(archive_events)
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                
                cols = ['datetime', 'trip_id', 'stop_sequence', 'stop_id', 'actual_arrival_delay', 'arrival_uncertainty', 
                        'actual_departure_delay', 'departure_uncertainty', 'timestamp']
                df = df[cols]
                
                # Append to CSV, writing the header only once
                df.to_csv(OUTPUT_CSV_FILE, mode='a', header=write_header, index=False)
                total_events_found += len(archive_events)
                write_header = False # Disable header for subsequent appends
                
                print(f"  [SUCCESS] Finished writing. Total events written: {total_events_found}")
            else:
                 print("  No new completed events detected in this archive.")


        except Exception as e:
            print(f"ERROR: Failed to process archive {archive_path}. Reason: {e}")
        finally:
            # 5. Clean up the temporary directory regardless of success or failure
            print("  Cleaning up temporary files...")
            shutil.rmtree(temp_dir)

    # --- Finalize ---
    # The final dataset is already on disk.
    print(f"\nProcessing complete. Found a total of {total_events_found} unique stop events written to {OUTPUT_CSV_FILE}.")
    print("--- Pipeline Finished Successfully ---")


if __name__ == "__main__":
    main()

