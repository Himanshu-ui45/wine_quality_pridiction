
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained ML model
obj = joblib.load("wine_model.joblib")

model = obj["model"]
columns = obj["columns"]


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = []

        for column in columns:
            value = request.form.get(column)

            if value is None or value.strip() == "":
                return render_template(
                    "index.html",
                    prediction="Please enter all values."
                )

            values.append(float(value))

        input_data = np.array(values).reshape(1, -1)

        prediction = model.predict(input_data)[0]

        prediction = max(0, min(10, float(prediction)))
        prediction = round(prediction, 2)

        return render_template(
            "index.html",
            prediction=prediction
        )

    except (ValueError, TypeError) as e:
        return render_template(
            "index.html",
            prediction=f"Invalid input: {str(e)}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
