import streamlit as st
import joblib
import numpy as np
import sklearn

# Set page configuration
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

# App title and description
st.title("🚗 Car Price Predictor")
st.markdown("""
Welcome to the Car Price Predictor! Enter the **age of your car**, and we'll predict its price using our trained machine learning model.
""")

# Load the trained model
st.sidebar.header("Model Status")
try:
    model = joblib.load('car_price_predictor.pkl')
    st.sidebar.success("Model loaded successfully!")
except FileNotFoundError:
    st.sidebar.error("Model file not found. Please check the file path.")
    model = None
except Exception as e:
    st.sidebar.error(f"An error occurred while loading the model: {e}")
    model = None

# User input form
with st.form(key='predict_form'):
    st.subheader("Enter Car Details")
    age = st.text_input('Enter car age', value="5", max_chars=3)
    submit_button = st.form_submit_button(label="🚘 Predict Price")

if submit_button:
    st.divider()  # Adds a divider line
    if model:  # Ensure the model is loaded
        try:
            # Convert age to integer
            age = int(age)
            if age < 0:
                st.error("❌ Car age cannot be negative. Please enter a valid age.")
            else:
                # Prepare input data
                input_data = np.array([[age]])  # 2D array for prediction
                # Make prediction
                predicted_price = model.predict(input_data)[0]
                # Display result
                st.success(f"✅ Predicted Price: **Rs. {predicted_price:.2f}**")
        except ValueError:
            st.error("❌ Please enter a valid numeric value for car age.")
        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {e}")
    else:
        st.error("❌ The model is not loaded. Please check the setup.")

# Footer
st.divider()
st.markdown("""
Developed with ❤️ using [Streamlit](https://streamlit.io).  
For any issues, contact [Support](mailto:support@example.com).
""")
