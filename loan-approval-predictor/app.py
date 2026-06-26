import streamlit as st
import pandas as pd

from src.preprocess import preprocess_input
from src.predict import predict_loan

st.set_page_config(
    page_title="Smart Loan Approval System",
    page_icon="🏦",
    layout="wide"
)

st.markdown("""
<style>

.main {
    background-color:#f8fafc;
}

.loan-header{
    background:linear-gradient(90deg,#0f172a,#1e40af);
    padding:30px;
    border-radius:15px;
    color:white;
    text-align:center;
    margin-bottom:20px;
}

.stButton>button{
    width:100%;
    height:55px;
    border-radius:10px;
    font-size:18px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="loan-header">
<h1>🏦 Smart Loan Approval System</h1>
<h4>AI Powered Banking Decision Platform</h4>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("🏦 Loan Portal")

st.sidebar.success(
    "AI Based Loan Approval"
)

st.sidebar.markdown("""
### Features

✔ Loan Prediction

✔ Approval Probability

✔ Risk Assessment

✔ Eligibility Score

✔ EMI Calculator
""")

st.header("👤 Applicant Information")

col1, col2 = st.columns(2)

with col1:

    no_of_dependents = st.selectbox(
        "Number of Dependents",
        [0,1,2,3,4,5]
    )

    education = st.selectbox(
        "Education",
        ["Graduate","Not Graduate"]
    )

with col2:

    self_employed = st.selectbox(
        "Self Employed",
        ["No","Yes"]
    )

st.header("💰 Financial Information")

col3, col4 = st.columns(2)

with col3:

    income_annum = st.number_input(
        "Annual Income (₹)",
        min_value=0,
        value=500000
    )

    loan_amount = st.number_input(
        "Loan Amount (₹)",
        min_value=0,
        value=1000000
    )

with col4:

    loan_term = st.number_input(
        "Loan Term (Months)",
        min_value=1,
        value=12
    )

    cibil_score = st.slider(
        "CIBIL Score",
        300,
        900,
        700
    )

st.header("🏠 Assets")

col5, col6 = st.columns(2)

with col5:

    residential_assets_value = st.number_input(
        "Residential Assets",
        min_value=0,
        value=5000000
    )

    commercial_assets_value = st.number_input(
        "Commercial Assets",
        min_value=0,
        value=2000000
    )

with col6:

    luxury_assets_value = st.number_input(
        "Luxury Assets",
        min_value=0,
        value=3000000
    )

    bank_asset_value = st.number_input(
        "Bank Assets",
        min_value=0,
        value=1000000
    )

total_assets = (
    residential_assets_value +
    commercial_assets_value +
    luxury_assets_value +
    bank_asset_value
)

st.markdown("---")

colA,colB,colC = st.columns(3)

with colA:
    st.metric(
        "Income",
        f"₹{income_annum:,}"
    )

with colB:
    st.metric(
        "Loan Amount",
        f"₹{loan_amount:,}"
    )

with colC:
    st.metric(
        "Total Assets",
        f"₹{total_assets:,}"
    )

st.markdown("---")

if st.button(
    "🔍 Predict Loan Status",
    use_container_width=True
):

    input_df = preprocess_input(
        no_of_dependents,
        education,
        self_employed,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
    )

    prediction, probability = predict_loan(input_df)

    result = "Approved" if prediction == 0 else "Rejected"

    approval_probability = probability[0] * 100

    st.subheader("📊 Loan Assessment")

    st.progress(int(approval_probability))

    st.metric(
        "Approval Probability",
        f"{approval_probability:.2f}%"
    )

    if cibil_score >= 750:
        risk = "🟢 Low Risk"

    elif cibil_score >= 650:
        risk = "🟡 Medium Risk"

    else:
        risk = "🔴 High Risk"

    st.info(
        f"Risk Category: {risk}"
    )

    score = min(
        100,
        round(approval_probability)
    )

    st.metric(
        "Eligibility Score",
        f"{score}/100"
    )

    if result == "Approved":

        st.success(
            "✅ Congratulations! Loan Approved"
        )

        st.balloons()

    else:

        st.error(
            "❌ Loan Rejected"
        )

st.markdown("---")

st.header("💳 EMI Calculator")

rate = st.number_input(
    "Interest Rate (%)",
    value=8.5
)

years = st.number_input(
    "Loan Tenure (Years)",
    value=5
)

monthly_rate = rate/(12*100)

months = years*12

emi = (
    loan_amount *
    monthly_rate *
    (1+monthly_rate)**months
)/(
    (1+monthly_rate)**months -1
)

st.metric(
    "Estimated EMI",
    f"₹{emi:,.2f}"
)

st.markdown("---")

st.caption(
    "Model: Random Forest Classifier | Loan Approval Prediction System"
)