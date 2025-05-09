import joblib

# Load joblib file dari model
model = joblib.load("lgbm_model.joblib")


def predict_data(transform_data):
    # Prediksi model: result
    result = model.predict(transform_data)

    # Mengembalikan hasil prediksi
    return result
