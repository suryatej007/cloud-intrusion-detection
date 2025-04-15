import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load models and threshold
scaler = joblib.load("models/scaler.pkl")
rf_model = joblib.load("models/random_forest.pkl")
autoencoder = load_model("models/autoencoder.h5", compile=False)
threshold = joblib.load("models/autoencoder_threshold.pkl")

# Preprocess user input
def preprocess_input(user_input):
    arr = np.array(user_input).reshape(1, -1)
    return scaler.transform(arr)

# Prediction using combined decision
def predict_combined(scaled_input):
    # Autoencoder Prediction
    recon = autoencoder.predict(scaled_input)
    recon_error = np.mean(np.square(scaled_input - recon), axis=1)[0]
    ae_score = np.clip(recon_error / threshold, 0, 1)  # Normalized score
    ae_pred = int(recon_error > threshold)

    # Random Forest Prediction
    rf_prob = rf_model.predict_proba(scaled_input)[:, 1][0]  # Probability of attack
    rf_pred = rf_model.predict(scaled_input)[0]

    # Weighted Combination (alpha for Random Forest, (1-alpha) for Autoencoder)
    alpha = 0.5  # Weight for Random Forest model
    combined_prob = alpha * rf_prob + (1 - alpha) * ae_score
    final_pred = int(combined_prob > 0.5)  # Final classification (Attack/Normal)

    # Return final prediction and debug info
    result = "Attack Detected" if final_pred == 1 else "Normal Traffic"

    debug_info = {
        "rf_pred": rf_pred,
        "rf_prob": rf_prob,
        "ae_pred": ae_pred,
        "ae_score": ae_score,
        "recon_error": recon_error,
        "threshold": threshold,
        "combined_prob": combined_prob
    }

    return result, debug_info
