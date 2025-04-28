# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load your dataset
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    else:
        return None

# Main function
def main():
    st.set_page_config(page_title="Breast Cancer Prediction App", page_icon="🩺", layout="wide")
    
    st.title("🩺 Breast Cancer Prediction using LightGBM")
    st.markdown("Predict whether a tumor is **benign** or **malignant** based on medical features.")
    st.write("---")
    
    # File uploader for user to upload data
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Load Data
        df = load_data(uploaded_file)
        
        if df is not None:
            # Sidebar for user input
            st.sidebar.header("Input Features")
            
            feature_columns = [col for col in df.columns if col != 'diagnosis']
            X = df[feature_columns]
            y = df['diagnosis']
            
            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Model
            model = LGBMClassifier()
            model.fit(X_train, y_train)
            
            # Sidebar Inputs
            user_input = {}
            for feature in feature_columns:
                user_input[feature] = st.sidebar.slider(
                    label=feature.replace('_', ' ').capitalize(),
                    min_value=float(X[feature].min()),
                    max_value=float(X[feature].max()),
                    value=float(X[feature].mean())
                )
            
            # Prediction
            user_df = pd.DataFrame([user_input])
            prediction = model.predict(user_df)[0]
            prediction_proba = model.predict_proba(user_df)[0]
            
            # Display Prediction
            st.subheader("Prediction Result")
            if prediction == 0:
                st.success(f" The tumor is predicted to be **Benign** with probability {prediction_proba[0]:.2%}.")
            else:
                st.error(f" The tumor is predicted to be **Malignant** with probability {prediction_proba[1]:.2%}.")
    
            st.write("---")
            
            # Show model performance
            st.subheader("Model Performance on Test Set")
            y_test_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_test_pred)
            st.metric(label="Accuracy", value=f"{acc:.2%}")
            
            st.write("---")
            
            # Optional: Show full dataset
            with st.expander("🔎 View Raw Data"):
                st.dataframe(df)
        else:
            st.error("There was an issue loading the dataset. Please check the file format and try again.")
    else:
        st.warning("Please upload a CSV file to proceed.")
    

if __name__ == "__main__":
    main()
