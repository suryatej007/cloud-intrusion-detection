import numpy as np
import joblib
from tensorflow.keras.models import load_model

# === Load models and threshold ===
scaler = joblib.load("models/scaler.pkl")
rf_model = joblib.load("models/random_forest.pkl")
autoencoder = load_model("models/autoencoder.h5", compile=False)
threshold = float(joblib.load("models/autoencoder_threshold.pkl"))  # Ensure it's a float

# === Preprocess input ===
def preprocess_input(user_input):
    arr = np.array(user_input).reshape(1, -1)
    return scaler.transform(arr)

# === Prediction logic ===
def predict_combined(scaled_input, alpha=0.7):
    # Autoencoder Prediction
    recon = autoencoder.predict(scaled_input, verbose=0)
    recon_error = np.mean(np.square(scaled_input - recon), axis=1)[0]
    ae_score = np.clip(recon_error / threshold, 0, 1)
    ae_pred = int(recon_error > threshold)

    # Random Forest Prediction
    rf_pred = int(rf_model.predict(scaled_input)[0])
    rf_prob = float(rf_model.predict_proba(scaled_input)[:, 1][0])

    # Combined Weighted Score
    combined_prob = alpha * rf_prob + (1 - alpha) * ae_score
    combined_pred = int(combined_prob > 0.5)

    result = "Attack Detected" if combined_pred == 1 else "Normal Traffic"

    debug_info = {
        "rf_pred": rf_pred,
        "rf_prob": rf_prob,
        "ae_pred": ae_pred,
        "recon_error": float(recon_error),
        "threshold": float(threshold),
        "ae_score": float(ae_score),
        "combined_prob": float(combined_prob),
        "combined_pred": combined_pred
    }

    return result, debug_info
