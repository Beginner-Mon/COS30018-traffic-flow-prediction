from flask import Flask, jsonify, request, render_template

app = Flask(__name__)



@app.route('/api/data', methods=['POST'])
def post_data():
    data = request.json  # Get JSON data from the request
    return jsonify({'received': data}), 201  # Return the received data

if __name__ == '__main__':
    app.run(debug=True)  # Run the app in debug mode