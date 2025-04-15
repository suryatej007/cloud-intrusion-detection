
import streamlit as st
from utils import preprocess_input, predict_combined

st.title("Cloud Intrusion Detection (Combined Model)")
st.markdown("Enter 5 feature values:")

f1 = st.number_input("duration", value=0.0)
f2 = st.number_input("src_bytes", value=0.0)
f3 = st.number_input("dst_bytes", value=0.0)
f4 = st.number_input("count", value=0.0)
f5 = st.number_input("serror_rate", value=0.0)

if st.button("Predict"):
    user_input = [f1, f2, f3, f4, f5]
    scaled = preprocess_input(user_input)
    result, debug = predict_combined(scaled)

    st.success(result)

    # Show detailed debug info
    st.markdown("### 🔍 Model Insights")
    st.write(f"Random Forest Prediction: {'Attack' if debug['rf_pred'] == 1 else 'Normal'}")
    st.write(f"Autoencoder Reconstruction Error: {debug['recon_error']:.6f}")
    st.write(f"Autoencoder Threshold: {debug['threshold']:.6f}")
