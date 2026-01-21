import streamlit as st
import pandas as pd
import pickle

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

# --- LOAD MODEL & SCALER ---
@st.cache_resource
def load_assets():
    model = pickle.load(open('model_churn_rf.pkl', 'rb'))
    scaler = pickle.load(open('scaler_churn.pkl', 'rb'))
    features = pickle.load(open('feature_columns.pkl', 'rb')) 
    return model, scaler, features

try:
    model, scaler, feature_columns = load_assets()
except:
    st.error("Error: Make sure all files are available.")

# --- SIDEBAR: CUSTOMER INPUT ---
st.sidebar.header("📋 Customer Profile")

def user_input_features():
    # Fitur Numerik
    tenure = st.sidebar.slider("Tenure", 0, 72, 12)
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 0.0, 150.0, 70.0)
    total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 8000.0, 800.0)
    
    # Fitur Kategorikal
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
    dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
    
    st.sidebar.subheader("Services & Contracts")
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.sidebar.selectbox("Internet Services", ["DSL", "Fiber optic", "No"])
    pay_method = st.sidebar.selectbox("Payment Method", 
                                     ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    
    # Save to Dictionary
    data = {
        'tenure': tenure, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges,
        'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
        'Contract': contract, 'InternetService': internet, 'PaymentMethod': pay_method
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# --- MAIN DASHBOARD ---
st.title("🛡️ Telco Retention Strategy Dashboard")
st.markdown("""
This application uses a **Random Forest** model that has been optimized to detect customers at risk of churn.
""")

# --- PREPROCESSING INPUT ---
# Match the input to the format during training (One-Hot Encoding)
# Note: Use dummy encoding that is consistent with the training data.
input_processed = pd.get_dummies(input_df)
input_processed = input_processed.reindex(columns=feature_columns, fill_value=0)

# Scaling numerik
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
input_processed[num_cols] = scaler.transform(input_processed[num_cols])

# --- PREDICTION ---
prediction = model.predict(input_processed)
prediction_proba = model.predict_proba(input_processed)[0][1]

# --- DISPLAY RESULTS ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Analysis Results")
    risk_score = f"{prediction_proba:.1%}"
    
    if prediction[0] == 1:
        st.error(f"**HIGH RISK: CHURN**")
        st.metric(label="Churn Probability", value=risk_score, delta="High Risk")
    else:
        st.success(f"**LOW RISK: LOYAL**")
        st.metric(label="Churn Probability", value=risk_score, delta="- Low Risk", delta_color="inverse")

    # Visualisasi Gauge Sederhana
    st.progress(prediction_proba)

with col2:
    st.subheader("Recommendation Strategy")
    if prediction_proba > 0.7:
        st.warning("⚠️ **Immediate Action Required:** Offer a 1 year contract discount or a free 3 month service upgrade.")
    elif 0.4 < prediction_proba <= 0.7:
        st.info("ℹ️ **Engagement Strategy:** Send out satisfaction questionnaires and offer premium technical support.")
    else:
        st.success("✅ **Retention Strategy:** Provide standard loyalty rewards (newsletter or points).")

# --- CUSTOMER PROFILE DETAILS ---
st.divider()
st.subheader("🔍 Data Profiling")
st.table(input_df)

# --- FOOTER ---
st.caption("Developed by Auliya Rohmah | Model: Optimized Random Forest")