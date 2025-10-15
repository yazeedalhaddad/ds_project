from flask import Flask, request, jsonify, render_template
from trafiklab_sim.predict_delay_v2 import DelayPredictorV2
import pandas as pd
import numpy as np

# Initialize the Flask app in the correct location
# It needs to know where the 'templates' folder is.
app = Flask(__name__)

# --- Load the AI Model ---
try:
    predictor = DelayPredictorV2("trafiklab_sim/model_artifacts_v2")
    print("AI Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    predictor = None

# --- CORRECTED: Load GTFS data for UI dropdowns ---
try:
    # This block now correctly loads data from your files
    routes_df = pd.read_csv("trafiklab_sim/gtfs_static/routes.txt")
    stops_df = pd.read_csv("trafiklab_sim/gtfs_static/stops.txt")

    # Create the lists and mappings for the application
    ROUTE_OPTIONS = sorted(routes_df['route_short_name'].unique().tolist())
    STOP_OPTIONS = sorted(stops_df['stop_name'].unique().tolist())

    # These maps translate the user-friendly names back to the IDs the model needs
    STOP_ID_MAP = dict(zip(stops_df['stop_name'], stops_df['stop_id']))
    ROUTE_ID_MAP = dict(zip(routes_df['route_short_name'], routes_df['route_id']))
    print("Successfully loaded route and stop names for dropdowns.")
except Exception as e:
    # If this fails, the lists will be empty, and we'll see an error message in the terminal
    print(f"CRITICAL ERROR: Could not load GTFS files for UI options: {e}")
    ROUTE_OPTIONS = []
    STOP_OPTIONS = []
    STOP_ID_MAP = {}
    ROUTE_ID_MAP = {}

# --- Web Page Route ---
@app.route('/')
def home():
    """Renders the main HTML page."""
    # This now correctly passes the loaded lists to your HTML file
    return render_template('trip_delay_planner_single_page_app-5.html',
                           routes=ROUTE_OPTIONS,
                           stops=STOP_OPTIONS)

# --- Prediction API Route ---
@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to get a delay prediction."""
    if predictor is None:
        return jsonify({"error": "Model is not loaded."}), 500

    try:
        data = request.json
        print("Received prediction request:", data)

        # Prepare data for the model
        departure_time = pd.to_datetime(data['departure_time'])
        departure_hour = departure_time.hour
        dow = departure_time.dayofweek
        is_weekend = 1 if dow >= 5 else 0

        # Translate names from dropdowns to IDs
        route_id = ROUTE_ID_MAP.get(data['route'])
        origin_stop_id = STOP_ID_MAP.get(data['start_stop'])
        dest_stop_id = STOP_ID_MAP.get(data['dest_stop'])

        if not all([route_id, origin_stop_id, dest_stop_id]):
             return jsonify({"error": "Invalid route or stop name provided."}), 400

        # Create the weather dictionary
        weather = {
            "state": data['weather_state'],
            "temp_c": float(data['temp_c']),
            "wind_ms": float(data['wind_ms']),
            "visibility_km": float(data['visibility_km']),
            "is_rain": 1 if data['weather_state'] == 'rain' else 0,
            "is_snow": 1 if data['weather_state'] == 'snow' else 0,
            "dow": dow
        }

        # Call the AI predictor
        delay_sec = predictor.predict_between_stops(
            route_id=route_id,
            origin_stop_id=origin_stop_id,
            dest_stop_id=dest_stop_id,
            departure_hour=departure_hour,
            weather=weather,
            origin_departure_delay_sec=0,
            headway_min=15,
            dow=dow,
            is_weekend=is_weekend
        )

        return jsonify({'predicted_delay_seconds': round(delay_sec, 1)})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 400

# --- Main entry point to run the server ---
if __name__ == '__main__':
    app.run(debug=True, port=5001)
