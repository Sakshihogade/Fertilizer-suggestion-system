from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Importing pickle files
model = pickle.load(open('classifier.pkl', 'rb'))
ferti = pickle.load(open('fertilizer (1).pkl', 'rb'))

@app.route('/')
def home():
    return jsonify({"message": "Fertilizer Suggestion API is running!"})


@app.route('/predict', methods=['POST'])
def predict():
    # Check if Content-Type is application/json
    if request.content_type != 'application/json':
        return jsonify({"error": "Invalid Content-Type. Expected 'application/json'."}), 415

    # Parse the JSON data from the request
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON data in request body."}), 400

    # Extract values from JSON
    temp = data.get('temp')
    humi = data.get('humid')
    mois = data.get('mois')
    soil = data.get('soil')
    crop = data.get('crop')
    nitro = data.get('nitro')
    pota = data.get('pota')
    phosp = data.get('phos')

    # Check for missing or invalid values
    if None in (temp, humi, mois, soil, crop, nitro, pota, phosp):
        return jsonify({"error": "Missing or invalid input values."}), 400

    try:
        # Convert inputs to integers and create input array
        input_features = np.array(
            [[int(temp), int(humi), int(mois), int(soil), int(crop), int(nitro), int(pota), int(phosp)]]
        )
    except ValueError:
        return jsonify({"error": "Invalid input values. All inputs must be integers."}), 400

    try:
        # Make a prediction
        prediction = model.predict(input_features)  # Returns an ndarray
        result = ferti.classes_[prediction][0]  # Get the class label for prediction
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

    # Return the result as JSON
    return jsonify({"prediction": result})


if __name__ == "__main__":
    app.run(debug=True)
