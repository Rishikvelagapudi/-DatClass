import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

def load_model():
    try:
        model = joblib.load('saved_model/classifier.pkl')
        scaler = joblib.load('saved_model/scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None

def main():
    st.set_page_config(page_title="Iris Species Predictor", layout="wide")
    st.title("🌺 Iris Data Classification Web App")
    st.markdown("Predict the species of an Iris flower based on its sepal and petal dimensions.")

    model, scaler = load_model()

    if model is None or scaler is None:
        st.error("Model or Scaler not found! Please run `train.py` first to train and save the model.")
        return

    tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction (CSV)", "Model Metrics"])

    with tab1:
        st.header("Manual Input Prediction")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
                sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
            with col2:
                petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
                petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

            submitted = st.form_submit_button("Predict Species")
            
            if submitted:
                features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
                features_scaled = scaler.transform(features)
                prediction = model.predict(features_scaled)
                st.success(f"The predicted species is: **{prediction[0].upper()}**")

    with tab2:
        st.header("Batch Prediction")
        st.markdown("Upload a CSV file containing `sepal_length`, `sepal_width`, `petal_length`, and `petal_width` columns.")
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(df.head())
            
            # Simple check for required columns
            required_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            # Fallback to column indices if names do not match perfectly
            if len(df.columns) >= 4:
                features = df.iloc[:, :4].values
                features_scaled = scaler.transform(features)
                predictions = model.predict(features_scaled)
                
                df['predicted_species'] = predictions
                st.write("Prediction Results:")
                st.dataframe(df)
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv',
                )
            else:
                st.error("The uploaded CSV must contain at least 4 numerical columns for features.")

    with tab3:
        st.header("Model Evaluation Metrics")
        st.markdown(f"**Current Model Type:** {type(model).__name__}")
        
        if os.path.exists('accuracy_comparison.png'):
            st.image('accuracy_comparison.png', caption='Algorithm Comparison')
        elif os.path.exists('images/accuracy_comparison.png'):
            st.image('images/accuracy_comparison.png', caption='Algorithm Comparison')
        else:
            st.warning("Comparison image not found. Run train.py to generate it.")

if __name__ == "__main__":
    main()
