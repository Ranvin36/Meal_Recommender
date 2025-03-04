from flask import Flask,request,jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_cors import CORS, cross_origin
import pandas as pd
import joblib

def create_app(test_config=None):
    app = Flask(__name__,instance_relative_config=True)
    CORS(app, resources={r"*": {"origins": "*"}})
    app.config['CORS_HEADERS'] = 'Content-Type'
    model = joblib.load("./Dataset/knn.pkl")
    dataset = pd.read_csv("./Dataset/preprocessed_dataset.csv")
    tfid = joblib.load("./SavedModels/tfid_vectorizer.pkl")

    @app.route("/test",methods=["POST"])
    def testEndpoint():
        data = request.get_json()
        print("Request Received",data)
        return jsonify(data)

    @app.route("/predict",methods=["POST"])
    def predict():
        data = request.get_json()
        join_ing = " ".join(data['ingredients'])
        vector_ing = tfid.transform([join_ing])
        distance , indices = model.kneighbors(vector_ing,n_neighbors=10)
        recipe = dataset.iloc[indices[0]].to_dict(orient='records')
        return jsonify(recipe)
    return app