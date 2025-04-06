from flask import Flask, request, jsonify
import pandas as pd
from route_finding import find_top_routes, get_all_scats_data
from route_finding_stgnn import route_finding_stgnn  # Import the STGNN route finder
from flask_cors import CORS

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})  # Restrict to React origin

# Existing endpoint for find_routes
@app.route('/find_routes', methods=['POST'])
def find_routes():
    data = request.get_json()
    origin = data.get('source')
    destination = data.get('destination')

    if not origin or not destination:
        return jsonify({'error': 'Missing source or destination'}), 400

    routes, error = find_top_routes(origin, destination)
    print(routes, error)
    if error:
        return jsonify({'error': error}), 404
    return jsonify({'routes': routes}), 200

# Existing endpoint for SCATS data
@app.route('/get_scats_number', methods=['GET'])
def get_scats_number():
    scats = get_all_scats_data()
    return jsonify({
        'scats': scats
    }), 200
@app.route('/find_routes_stgnn', methods=['POST'])
def find_routes_stgnn():
    data = request.get_json()
    origin = data.get('source')
    destination = data.get('destination')
    start_time_str = data.get('start_time')

    if not origin or not destination or not start_time_str:
        return jsonify({'error': 'Missing source, destination, or start_time'}), 400

    try:
        origin = int(origin)
        destination = int(destination)
        pd.to_datetime(start_time_str, format='%d-%m-%Y %H:%M')
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400

    try:
        result = route_finding_stgnn(
            spatial_file="traffic_network2.csv",
            temporal_file="TrainingDataAdaptedOutput.csv",
            model_path="results/best_model.pt",
            start_scat=origin,
            dest_scat=destination,
            start_time_str=start_time_str,
            scaler_path="results/scaler.pt"
        )
        if not result['routes']:
            return jsonify({'error': 'No routes found between the specified SCATS IDs'}), 404
        return jsonify(result), 200
    except Exception as e:
        print(f"Error in find_routes_stgnn: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500
# Chạy server
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)