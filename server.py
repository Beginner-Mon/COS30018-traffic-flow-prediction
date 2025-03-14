# api.py
from flask import Flask, request, jsonify
from route_finding import find_top_routes
from flask_cors import CORS
# Khởi tạo Flask app
app = Flask(__name__)
CORS(app, resources={r"/find_routes": {"origins": "http://localhost:3000"}})  # Restrict to React origin
# API endpoint
@app.route('/find_routes', methods=['POST'])
def find_routes():
    # Lấy dữ liệu từ request
    data = request.get_json()
    origin = data.get('source')
    destination = data.get('destination')

    # Kiểm tra input cơ bản
    if not origin or not destination:
        return jsonify({'error': 'Missing source or destination'}), 400

    # Gọi hàm tính toán từ routefinding.py
    routes, error = find_top_routes(origin, destination)
    print (routes, error)
    # Xử lý kết quả
    if error:
        return jsonify({'error': error}), 404
    return jsonify({'routes': routes}), 200

# Chạy server
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)