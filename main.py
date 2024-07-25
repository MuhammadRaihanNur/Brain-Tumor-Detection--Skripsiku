from flask import Flask, render_template, send_from_directory, request, jsonify
import pandas as pd
import numpy as np 
import pickle

from sklearn.tree import DecisionTreeClassifier
# import time
# import logging
# import joblib
# import sklearn
app = Flask(__name__)

# Load model
# Load the trained model
# model = pickle.load(open('model.pkl', 'rb'))



# Load model_data.pkl
def load_model_data():
    try:
        with open('model_data.pkl', 'rb') as model_file:
            model_data = pickle.load(model_file)
        return model_data
    except Exception as e:
        print(f"Error loading model data: {str(e)}")
        return None

# Global variable to hold model data
model_data = load_model_data()

# logging.basicConfig(level=logging.DEBUG)

# def load_model(file_path):
#     try:
#         with open(file_path, 'rb') as f:
#             model = pickle.load(f)
#         print(f"Model from {file_path} loaded successfully.")
#         return model
#     except FileNotFoundError:
#         print(f"File not found: {file_path}")
#     except pickle.UnpicklingError:
#         print(f"Unpickling error: {file_path} may be corrupted or not a pickle file.")
#     except Exception as e:
#         print(f"An error occurred while loading the model from {file_path}: {e}")

# # Load models
# dt_model = load_model('dt.pkl')
# nb_model = load_model('nb.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Brain_Tumor.csv')
def download_file():
    return send_from_directory('dataset', 'Brain_Tumor.csv')

# Rute untuk halaman Accuracy
@app.route('/model')
def model():
    return render_template('model.html')

@app.route('/dataset')
def dataset():
    # # Baca file CSV
    # df = pd.read_csv('./dataset/Brain_Tumor.csv')

    # # Ambil kolom dan data
    # columns = df.columns.tolist()
    # data = df.to_dict(orient='records')

    # # Pastikan data tidak None
    # if data is None:
    #     data = []
    # if columns is None:
    #     columns = []

    # Render template dengan data
    return render_template('dataset.html')

# Rute untuk halaman Kontak
@app.route('/prediksi')
def prediksi():
    return render_template('predict.html')

@app.route('/tabel')
def table():
    df = pd.read_csv('./Brain_Tumor.csv')

    # Ambil kolom dan data
    columns = df.columns.tolist()
    data = df.to_dict(orient='records')

    return render_template('tabel.html', columns=columns, data=data)


# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.json

    # Check if model_data is loaded successfully
    if model_data is None:
        return jsonify({"error": "Model data not loaded properly. Please check server logs."}), 500

    # Extract necessary components from model_data
    dt_model = model_data['dt_model']
    nb_model = model_data['nb_model']
    scaler = model_data['scaler']
    final_pca_dt = model_data['final_pca_dt']
    final_pca_nb = model_data['final_pca_nb']

    required_keys = [
        "Mean", "Variance", "Standard Deviation", "Entropy", "Skewness", "Kurtosis",
        "Contrast", "Energy", "ASM", "Homogeneity", "Dissimilarity", "Correlation", "Coarseness"
    ]

    if not all(key in data for key in required_keys):
        return jsonify({"error": f"JSON must contain the following keys: {', '.join(required_keys)}"}), 400

    try:
        # Convert data to DataFrame to maintain feature names
        features = pd.DataFrame([data], columns=required_keys)

        # Scale the features
        features_scaled = scaler.transform(features)

        # Perform PCA based on selected model
        features_pca_dt = final_pca_dt.transform(features_scaled)
        features_pca_nb = final_pca_nb.transform(features_scaled)

        # Predict using both models
        pred_dt = dt_model.predict(features_pca_dt)
        pred_nb = nb_model.predict(features_pca_nb)

        # Determine which classifier to use based on accuracy
        if model_data['dt_accuracy'] > model_data['nb_accuracy']:
            final_pred = dt_model.predict(final_pca_dt.transform(features_scaled))
            model_used = "Decision Tree"
        else:
            final_pred = nb_model.predict(final_pca_nb.transform(features_scaled))
            model_used = "Naive Bayes"

        # Map predictions to labels
        labels = {0: "Non Tumor", 1: "Tumor"}
        response = {
            'Hasil_Prediksi_DT': labels[int(pred_dt[0])],
            'Hasil_Prediksi_NB': labels[int(pred_nb[0])],
            'Hasil_Akhir': labels[int(final_pred[0])],
            'Model_Yang': model_used
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)