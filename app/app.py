from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load model
package = joblib.load("./iris_model.pkl")
model = package["model"]
class_names = package["class_names"]

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        input_df = pd.DataFrame([data])  # Convert input to DataFrame

        prediction = model.predict(input_df)[0]
        class_name = class_names[prediction]

        return jsonify({
            "prediction": int(prediction),
            "class_name": class_name
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
