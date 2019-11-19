from flask import Flask,request,jsonify,render_template
from joblib import dump, load
import numpy as np

app = Flask(__name__)

model = load("lr_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    int_features = [float(x) for x in request.form.values()]
    #~ print(int_features)
    final_features = [np.array(int_features)]
    #~ print(final_features)
    prediction = model.predict(final_features)
    #~ print(prediction)
    
    if prediction == 1:
        output = "setosa"
    elif prediction == 2:
        output = "versicolor"
    elif prediction == 3:
        output = "virginica"
    else:
        output = "UNKOWN!!"
    return render_template('index.html',prediction_text='Given input belongs to following Flower Species---->> __%s__ '%(output))
    
if __name__ == "__main__":
    app.run(debug=True)