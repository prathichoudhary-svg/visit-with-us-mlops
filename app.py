import streamlit as st
import pandas as pd
import joblib
import os

# Configure Page App
st.set_page_config(
    page_title="Visit With Us - Travel Package Predictor",
    page_icon="✈️",
    layout="centered"
)

# Page Title and Description
st.title("✈️ Customer Travel Package Purchase Predictor")
st.write("""
This application uses our champion machine learning model to predict whether a customer is likely to buy 
the newly introduced **Wellness Tourism Package** based on customer demographic and interaction data.
""")

# Paths to Model artifact
MODEL_PATH = 'visit_with_us_mlops/models/best_model.pkl'
if not os.path.exists(MODEL_PATH):
    # Secondary fallback path for Streamlit Cloud deployment context
    MODEL_PATH = 'models/best_model.pkl'

@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading model from {path}: {e}")
        return None

pipeline = load_model(MODEL_PATH)

if pipeline is not None:
    st.success("🚀 Champion Prediction Pipeline successfully loaded!")
    
    st.header("📋 Enter Customer Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        duration_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=120, value=15)
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        number_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
        number_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
        product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
        
    with col2:
        preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Unmarried", "Divorced"])
        number_trips = st.number_input("Number of Trips", min_value=1, max_value=20, value=3)
        passport = st.selectbox("Owns Passport?", ["Yes", "No"])
        pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
        own_car = st.selectbox("Owns Car?", ["Yes", "No"])
        number_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=1)
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=150000, value=25000)

    # Transform input variables to match dataset specs
    passport_numeric = 1 if passport == "Yes" else 0
    own_car_numeric = 1 if own_car == "Yes" else 0
    
    # Convert inputs to Dataframe structure matching training set schema
    input_data = pd.DataFrame({
        'Age': [float(age)],
        'TypeofContact': [typeof_contact],
        'CityTier': [int(city_tier)],
        'DurationOfPitch': [float(duration_pitch)],
        'Occupation': [occupation],
        'Gender': [gender],
        'NumberOfPersonVisiting': [int(number_person_visiting)],
        'NumberOfFollowups': [float(number_followups)],
        'ProductPitched': [product_pitched],
        'PreferredPropertyStar': [float(preferred_property_star)],
        'MaritalStatus': [marital_status],
        'NumberOfTrips': [float(number_trips)],
        'Passport': [passport_numeric],
        'PitchSatisfactionScore': [int(pitch_satisfaction_score)],
        'OwnCar': [own_car_numeric],
        'NumberOfChildrenVisiting': [float(number_children_visiting)],
        'Designation': [designation],
        'MonthlyIncome': [float(monthly_income)]
    })
    
    if st.button("🔮 Predict Purchase Probability", type="primary"):
        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0][1]
        
        st.write("--- ")
        st.subheader("🎯 Prediction Results")
        
        if prediction == 1:
            st.error(f"🚨 Likely Buyer! (Purchase Probability: {probability:.2%})")
            st.write("**Recommendation**: Prioritize this prospect. This customer matches the high-probability buyer segments.")
        else:
            st.success(f"❌ Unlikely Buyer. (Purchase Probability: {probability:.2%})")
            st.write("**Recommendation**: Soft target only. This customer might not find immediate value in the Wellness package.")
else:
    st.warning("⚠️ Could not load the machine learning model. Check if 'best_model.pkl' is inside 'visit_with_us_mlops/models/' directory.")
