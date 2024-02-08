from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__, static_folder='Templates')

# Load the pre-trained model
model = joblib.load("predit\Model\Clustr_model.pkl")

def get_custom_result(cluster_group):
    if cluster_group == '0':
        return f"Cluster: {cluster_group}\nThis Customer is with moderate engagement, moderate mobile usage, and an average transaction revenue. Tailored strategies are recommended to enhance the experience and engagement. Customer contributes moderately to overall revenue."
    elif cluster_group == '1':
        return f"Cluster: {cluster_group}\nA high-value customer, showing low bounce rates, significant pageviews, and an average transaction revenue of $45,798,870. Likely to be a loyal customer with a strong interest. Personalized strategies should be recommended to enhance experience and engagement, as customers play a crucial role in contributing to overall revenue."
    elif cluster_group == '2':
        return f"Cluster: {cluster_group}\nThis customer has low engagement and no transaction revenue. Explore ways to improve engagement and encourage transactions are advised."
    elif cluster_group == '3':
        return f"Cluster: {cluster_group}\nA customer with low mobile usage, extremely high average visit numbers, and notable transaction revenue of $32,729,450 likely represents a specific user behavior pattern. These customers exhibit more loyalty, and personalized strategies can be implemented to maximize their experience and contribute even more to the overall revenue."
    else:
        return "Unknown cluster"

def predict_revenue(data):
    try:
        # Make prediction using the loaded model
        cluster_group = model.predict(data)[0]

        # Return the result
        return get_custom_result(str(cluster_group))

    except Exception as e:
        return str(e)

@app.route("/")
def index():
    return render_template("Cust_revenue_predit.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data and convert to a Pandas DataFrame
        form_data = {field: float(request.form[field]) for field in request.form}
        data = pd.DataFrame([form_data])

        # Get the prediction result
        result = predict_revenue(data)

        # Render the result on the result page
        return render_template("result.html", result=result)

    except Exception as e:
        return render_template("result.html", result=str(e))

if __name__ == "__main__":
    app.run(debug=True)
