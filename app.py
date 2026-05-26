# Enhanced `app.py` UI for Hotel Booking Intelligence System


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
# LOAD MODEL
# ==========================================

model = joblib.load('models/logistic_model.pkl')

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Import Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Main Background */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #1e293b 100%);
    color: white;
}

/* Remove Default Padding Issues */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Hero Section */
.hero-container {
    background: linear-gradient(135deg, rgba(37,99,235,0.35), rgba(139,92,246,0.30));
    padding: 2.5rem;
    border-radius: 24px;
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(12px);
    margin-bottom: 2rem;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.25);
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    color: white;
    margin-bottom: 0.5rem;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: #d1d5db;
    line-height: 1.7;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #0f172a 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] * {
    color: white;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 1.5rem;
    border-radius: 22px;
    backdrop-filter: blur(10px);
    box-shadow: 0px 8px 25px rgba(0,0,0,0.2);
    margin-bottom: 1.5rem;
}

/* Inputs */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
.stNumberInput div[data-baseweb="input"] {
    background-color: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    color: white !important;
}

.stTextInput input,
.stNumberInput input {
    color: white !important;
}

label {
    color: #f8fafc !important;
    font-weight: 500;
}

/* Radio Buttons */
.stRadio > div {
    background: rgba(255,255,255,0.04);
    padding: 0.8rem;
    border-radius: 14px;
}

/* Slider */
.stSlider {
    padding-top: 1rem;
}

/* Button */
.stButton > button {
    width: 100%;
    height: 3.6rem;
    border: none;
    border-radius: 18px;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    font-size: 1.05rem;
    font-weight: 700;
    transition: all 0.3s ease;
    box-shadow: 0px 6px 20px rgba(37,99,235,0.35);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0px 10px 25px rgba(124,58,237,0.4);
}

/* Metric Cards */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 1.4rem;
    border-radius: 22px;
    backdrop-filter: blur(10px);
    box-shadow: 0px 8px 25px rgba(0,0,0,0.2);
}

[data-testid="metric-container"] * {
    color: white !important;
}

/* Headers */
h1, h2, h3 {
    color: white !important;
}

/* Horizontal Line */
hr {
    border: none;
    height: 1px;
    background: rgba(255,255,255,0.1);
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🏨 Hotel Booking ML")

st.sidebar.markdown("---")

st.sidebar.markdown("""
### 📊 About This App

This AI-powered system predicts hotel booking cancellation probability using behavioral and operational booking patterns.

### ⚙️ ML Pipeline
- Logistic Regression
- Feature Engineering
- Threshold Optimization
- Class Imbalance Handling

### 🧠 Built With
- Python
- Scikit-learn
- Streamlit
- Pandas
""")

st.sidebar.markdown("---")

st.sidebar.success("🚀 Production Ready ML Application")

# ==========================================
# HERO SECTION
# ==========================================

st.markdown("""
<div class="hero-container">
    <div class="hero-title">🏨 Hotel Booking Intelligence System</div>
    <div class="hero-subtitle">
        Predict hotel booking cancellation risk using machine learning, behavioral analytics,
        and engineered business intelligence features.
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# INPUT SECTION
# ==========================================

st.markdown("## 📝 Booking Information")

col1, col2 = st.columns(2)

# ==========================================
# LEFT COLUMN
# ==========================================

with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

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

    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# RIGHT COLUMN
# ==========================================

with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

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

    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# SLIDER SECTION
# ==========================================

st.markdown('<div class="card">', unsafe_allow_html=True)

prev_cancel_ratio = st.slider(
    "📉 Previous Cancel Ratio",
    min_value=0.0,
    max_value=1.0,
    value=0.0
)

st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# FEATURE ENGINEERING
# ==========================================

room_changed = 1 if room_changed == 'Yes' else 0

high_risk_deposit = 1 if high_risk_deposit == 'High Risk' else 0

long_lead_booking = 1 if lead_time > 60 else 0

# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("🔮 Predict Cancellation"):

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

    st.markdown("---")

    st.markdown("## 📈 Prediction Results")

    result_col1, result_col2 = st.columns(2)

    # ======================================
    # PROBABILITY
    # ======================================

    with result_col1:

        st.metric(
            label="Cancellation Probability",
            value=f"{probability * 100:.2f}%"
        )

        st.progress(float(probability))

    # ======================================
    # FINAL RESULT
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

    st.markdown("## 🚦 Risk Assessment")

    if probability > 0.75:

        st.error("🔴 High Cancellation Risk")

    elif probability > 0.50:

        st.warning("🟠 Moderate Cancellation Risk")

    else:

        st.success("🟢 Low Cancellation Risk")

