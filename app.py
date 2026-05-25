
import streamlit as st
import pandas as pd
import joblib

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Hotel Booking Intelligence System",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Main App Background */
.main {
    background-color: #f5f7fa;
}

/* Title Styling */
h1 {
    color: #1f2937;
    font-weight: 800;
}

/* Headers */
h2, h3 {
    color: #374151;
}

/* Button Styling */
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: 600;
    border: none;
}

.stButton>button:hover {
    background-color: #1d4ed8;
    color: white;
}

/* Metric Cards */
[data-testid="metric-container"] {
    background-color: white;
    border: 1px solid #e5e7eb;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
}

/* Input Boxes */
.stNumberInput, .stSelectbox, .stSlider {
    background-color: white;
    border-radius: 10px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

section[data-testid="stSidebar"] * {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load('models/logistic_model.pkl')

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🏨 Hotel Booking ML")

st.sidebar.markdown("""
### 📊 About This App

This machine learning system predicts whether a hotel booking is likely to be canceled.

### ✅ Model Used
- Logistic Regression

### ⚙️ Features
- Feature Engineering
- Threshold Optimization
- Class Imbalance Handling
- Interactive Predictions

### 👨‍💻 Built With
- Python
- Scikit-learn
- Streamlit
""")

# ==========================================
# MAIN TITLE
# ==========================================

st.title("🏨 Hotel Booking Cancellation Intelligence")

st.markdown("""
### Predict booking cancellation risk using Machine Learning

This intelligent system analyzes hotel booking behavior and predicts
whether a reservation is likely to be canceled.
""")

st.markdown("---")

# ==========================================
# INPUT SECTION
# ==========================================

st.subheader("📝 Booking Information")

with st.container():

    col1, col2 = st.columns(2)

    # ======================================
    # LEFT COLUMN
    # ======================================

    with col1:

        lead_time = st.number_input(
            "📅 Lead Time",
            min_value=0,
            value=10
        )

        adr_per_night = st.number_input(
            "💰 ADR Per Night",
            min_value=0.0,
            value=100.0
        )

        total_nights = st.number_input(
            "🌙 Total Nights",
            min_value=1,
            value=2
        )

        total_special_requests = st.number_input(
            "⭐ Total Special Requests",
            min_value=0,
            value=1
        )

    # ======================================
    # RIGHT COLUMN
    # ======================================

    with col2:

        market_segment = st.selectbox(
            "📌 Market Segment",
            [
                'Online TA',
                'Offline TA/TO',
                'Direct',
                'Corporate',
                'Groups'
            ]
        )

        customer_type = st.selectbox(
            "👤 Customer Type",
            [
                'Transient',
                'Transient-Party',
                'Contract',
                'Group'
            ]
        )

        room_changed = st.radio(
            "🛏️ Was Room Changed?",
            ['No', 'Yes']
        )

        high_risk_deposit = st.radio(
            "⚠️ Deposit Risk",
            ['Low Risk', 'High Risk']
        )

# ==========================================
# SLIDER
# ==========================================

prev_cancel_ratio = st.slider(
    "📉 Previous Cancel Ratio",
    min_value=0.0,
    max_value=1.0,
    value=0.0
)

st.markdown("---")

# ==========================================
# CONVERT INPUTS
# ==========================================

room_changed = 1 if room_changed == 'Yes' else 0

high_risk_deposit = 1 if high_risk_deposit == 'High Risk' else 0

long_lead_booking = 1 if lead_time > 60 else 0

# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("🔮 Predict Cancellation"):

    # ======================================
    # CREATE INPUT DATAFRAME
    # ======================================

    input_data = pd.DataFrame({

        'prev_cancel_ratio': [prev_cancel_ratio],

        'long_lead_booking': [long_lead_booking],

        'high_risk_deposit': [high_risk_deposit],

        'lead_time': [lead_time],

        'adr_per_night': [adr_per_night],

        'market_segment': [market_segment],

        'room_changed': [room_changed],

        'total_of_special_requests': [total_special_requests],

        'total_nights': [total_nights],

        'customer_type': [customer_type]
    })

    # ======================================
    # MODEL PREDICTION
    # ======================================

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    # ======================================
    # RESULTS SECTION
    # ======================================

    st.markdown("---")

    st.header("📈 Prediction Results")

    result_col1, result_col2 = st.columns(2)

    # ======================================
    # PROBABILITY METRIC
    # ======================================

    with result_col1:

        st.metric(
            label="Cancellation Probability",
            value=f"{probability*100:.2f}%"
        )

        st.progress(float(probability))

    # ======================================
    # FINAL PREDICTION
    # ======================================

    with result_col2:

        if prediction == 1:

            st.metric(
                label="Prediction",
                value="Likely Canceled"
            )

        else:

            st.metric(
                label="Prediction",
                value="Likely Retained"
            )

    # ======================================
    # RISK INTERPRETATION
    # ======================================

    st.markdown("### 🚦 Risk Assessment")

    if probability > 0.75:

        st.error("🔴 High Cancellation Risk")

    elif probability > 0.50:

        st.warning("🟠 Moderate Cancellation Risk")

    else:

        st.success("🟢 Low Cancellation Risk")

