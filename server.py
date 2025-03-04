from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

@app.route('/submit', methods=['POST'])
def submit_data():
    data = request.json  # Get JSON data from the request
    lat = data.get('lat')
    long = data.get('long')
   
    
    # Here you can process the data (e.g., save to a database)
    
    return jsonify({
        'message': 'Data received successfully',
        'data': {
            'lat': lat,
            'long': long
        }
    })

if __name__ == '__main__':
    app.run(debug=True)