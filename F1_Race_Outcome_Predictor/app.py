import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="F1 Race Outcome Predictor",
    page_icon="🏎️",
    layout="wide"
)
st.image(
    "assets/f1_logo.png",
    use_container_width=True
)

st.title("🏎️ F1 Race Outcome Predictor")
st.markdown("""
### 🏎️ Dashboard | 📊 Analytics | 🏆 Models | 📈 Insights
""")

st.markdown("""
### Motorsport Analytics Dashboard

Predict whether an F1 driver will finish on the podium based on:

- Grid Position
- Qualifying Position
- Constructor
- Circuit
- Season Year

**Best Model:** Random Forest
""")
model = joblib.load(
    "models/podium_predictor.pkl"
)

constructor_encoder = joblib.load(
    "models/constructor_encoder.pkl"
)

circuit_encoder = joblib.load(
    "models/circuit_encoder.pkl"
)
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Dataset Rows",
        "26,759"
    )

with col2:
    st.metric(
        "Best F1 Score",
        "0.511"
    )

with col3:
    st.metric(
        "Best Model",
        "Random Forest"
    )

st.sidebar.header("🏁 Race Inputs")


grid = st.sidebar.number_input(
    "Grid Position",
    min_value=1,
    max_value=30,
    value=5
)

qualifying_position = st.sidebar.number_input(
    "Qualifying Position",
    min_value=1,
    max_value=30,
    value=5
)

constructor = st.sidebar.selectbox(
    "Constructor",
    constructor_encoder.classes_
)

circuit = st.sidebar.selectbox(
    "Circuit",
    circuit_encoder.classes_
)

year = st.sidebar.number_input(
    "Season Year",
    min_value=1950,
    max_value=2030,
    value=2024
)
constructor_encoded = (
    constructor_encoder.transform(
        [constructor]
    )[0]
)

circuit_encoded = (
    circuit_encoder.transform(
        [circuit]
    )[0]
)
if st.button("Predict"):

    input_data = pd.DataFrame({
        'grid':[grid],
        'qualifying_position':[qualifying_position],
        'constructor':[constructor_encoded],
        'circuit':[circuit_encoded],
        'year':[year]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success("🏆 Predicted Podium Finish")
    else:
        st.error("❌ Predicted Non-Podium Finish")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Podium Probability",
            f"{probability*100:.2f}%"
        )

    with col2:
        st.metric(
            "Prediction",
            "Podium" if prediction == 1 else "Non-Podium"
        )

st.subheader("📊 Random Forest Feature Importance")

importance_df = pd.DataFrame({
    'Feature':[
        'grid',
        'year',
        'circuit',
        'constructor',
        'qualifying_position'
    ],
    'Importance':[
        0.425822,
        0.183541,
        0.168746,
        0.121819,
        0.100072
    ]
})

fig, ax = plt.subplots(figsize=(8,4))

ax.bar(
    importance_df['Feature'],
    importance_df['Importance']
)

ax.set_title("Feature Importance")
ax.set_ylabel("Importance Score")

st.pyplot(fig)

importance_df = pd.DataFrame({
    'Feature':[
        'grid',
        'year',
        'circuit',
        'constructor',
        'qualifying_position'
    ],
    'Importance':[
        0.425822,
        0.183541,
        0.168746,
        0.121819,
        0.100072
    ]
})


st.subheader("🏆 Model Comparison")

results = pd.DataFrame({
    'Model':[
        'Logistic Regression',
        'Balanced Logistic Regression',
        'Random Forest',
        'XGBoost'
    ],
    'Accuracy':[
        0.886,
        0.741,
        0.859,
        0.892
    ],
    'F1 Score':[
        0.330,
        0.449,
        0.511,
        0.480
    ]
})

st.dataframe(results)

# ===========================
# THEN CSS
# ===========================

st.markdown("""
<style>

/* Main page */
.main {
    background-color: #0f1117;
}

/* Main title */
h1 {
    color: #E10600;
    text-align: center;
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #1a1a1a;
}

/* Sidebar header (Race Inputs) */
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h3 {
    color: #E10600 !important;
}

/* Labels: Grid Position, Constructor, Circuit, etc. */
[data-testid="stSidebar"] label {
    color: #FFFFFF !important;
    font-size: 18px !important;
    font-weight: 600 !important;
}

/* Metrics cards */
div[data-testid="metric-container"] {
    border: 1px solid #E10600;
    padding: 10px;
    border-radius: 10px;
}

/* Selectbox text */
.stSelectbox div[data-baseweb="select"] {
    color: #000000;
}

/* Number input text */
.stNumberInput input {
    color: #000000;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)