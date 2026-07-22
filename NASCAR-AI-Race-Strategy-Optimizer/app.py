import streamlit as st
import pandas as pd
from pathlib import Path

from src.predict import predict_finish
from src.strategy_engine import generate_strategy

st.set_page_config(
    page_title="NASCAR AI Race Strategy Optimizer",
    page_icon="🏁",
    layout="wide"
)

st.markdown("""
<div style="
background: linear-gradient(90deg,#D71920,#111111);
padding:20px;
border-radius:15px;
">

<h1 style="
color:white;
text-align:center;
margin-bottom:0px;
">
🏁 NASCAR AI Race Strategy Optimizer
</h1>

<p style="
color:white;
text-align:center;
font-size:18px;
margin-top:5px;
">
Machine Learning Powered Race Prediction & Strategy Analysis
</p>

</div>
""", unsafe_allow_html=True)


PROJECT_ROOT = Path(__file__).resolve().parent

BANNER = PROJECT_ROOT / "assets" / "banner.jpg"



DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_engineered.parquet"
)

df = pd.read_parquet(DATA_PATH)


st.markdown(
    """
# 🏁 NASCAR AI Race Strategy Optimizer

### Machine Learning Powered Race Prediction & Strategy Analysis
"""
)

BANNER_PATH = PROJECT_ROOT / "assets" / "banner.png"

st.sidebar.image("assets/banner.png", width=250)

st.sidebar.markdown("---")

st.sidebar.header("Race Configuration")

driver = st.sidebar.selectbox(
    "Driver",
    sorted(df["Driver"].unique())
)

track = st.sidebar.selectbox(
    "Track",
    sorted(df["Track"].unique())
)

team = st.sidebar.selectbox(
    "Team",
    sorted(df["Team"].unique())
)

make = st.sidebar.selectbox(
    "Manufacturer",
    sorted(df["Make"].unique())
)

series = st.sidebar.selectbox(
    "Series",
    sorted(df["Series"].unique())
)

start = st.sidebar.slider(
    "Starting Position",
    min_value=1,
    max_value=40,
    value=10
)

predict_button = st.sidebar.button(
    "Predict Race Strategy"
)

driver_data = df[df["Driver"] == driver]

latest = driver_data.iloc[-1]

input_data = pd.DataFrame({

    "Season": [latest["Season"]],
    "Race": [latest["Race"]],
    "Track": [track],
    "Length": [latest["Length"]],
    "Surface": [latest["Surface"]],
    "Start": [start],
    "Car": [latest["Car"]],
    "Driver": [driver],
    "Make": [make],
    "Team": [team],
    "Series": [series],
    "Laps": [latest["Laps"]],

    "Driver_Experience": [latest["Driver_Experience"]],
    "Driver_Historical_Avg_Finish": [latest["Driver_Historical_Avg_Finish"]],
    "Driver_Last5_Avg_Finish": [latest["Driver_Last5_Avg_Finish"]],
    "Driver_Historical_Avg_Rating": [latest["Driver_Historical_Avg_Rating"]],
    "Team_Historical_Avg_Finish": [latest["Team_Historical_Avg_Finish"]],
    "Make_Historical_Avg_Finish": [latest["Make_Historical_Avg_Finish"]],
    "Track_Historical_Avg_Finish": [latest["Track_Historical_Avg_Finish"]],

    "Track_Type": [latest["Track_Type"]],
    "Start_Category": [
        "Front" if start <= 5
        else "Mid" if start <= 15
        else "Back"
    ]

})

if predict_button:

    # ----------------------------------
    # Make Prediction
    # ----------------------------------

    prediction = predict_finish(input_data)

    # Rating

    if prediction <= 5:
        rating = "Excellent"
    elif prediction <= 10:
        rating = "Good"
    elif prediction <= 20:
        rating = "Average"
    else:
        rating = "Poor"

    confidence = max(50, min(99, int(100 - prediction)))

    # ----------------------------------
    # Top Metrics
    # ----------------------------------
    score = max(0, min(100, int((40 - prediction) / 40 * 100)))

    st.metric(
        "🏆 Strategy Score",
        f"{score}/100"
    )
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🏁 Predicted Finish", f"{prediction:.2f}")

    with col2:
        st.metric("⭐ Driver Rating", rating)

    with col3:
        st.metric("🎯 Confidence", f"{confidence}%")

    # ----------------------------------
    # Driver Profile
    # ----------------------------------

    st.markdown("---")
    st.subheader("👤 Driver Profile")

    left, right = st.columns(2)

    with left:
        st.write(f"**Driver:** {driver}")
        st.write(f"**Manufacturer:** {make}")
        st.write(f"**Series:** {series}")

    with right:
        st.write(f"**Team:** {team}")
        st.write(f"**Track:** {track}")
        st.write(f"**Starting Position:** {start}")

    # ----------------------------------
    # Prediction
    # ----------------------------------

    if prediction <= 5:
        st.success("🏆 Expected Top 5 Finish")

    elif prediction <= 10:
        st.info("🥈 Expected Top 10 Finish")

    elif prediction <= 20:
        st.warning("⚠️ Mid-Pack Finish Expected")

    else:
        st.error("🚩 Difficult Race Expected")
    # ----------------------------------
    # Strategy
    # ----------------------------------

    strategy_input = {
        "Start": start,
        "Driver_Experience": latest["Driver_Experience"],
        "Driver_Historical_Avg_Finish": latest["Driver_Historical_Avg_Finish"],
        "Team_Historical_Avg_Finish": latest["Team_Historical_Avg_Finish"],
        "Track_Type": latest["Track_Type"],
    }

    strategy = generate_strategy(strategy_input, prediction)

    st.subheader("📊 AI Strategy Recommendations")

    for item in strategy:

        st.info(f"🏁 {item}")

    # ----------------------------------
    # Driver Statistics
    # ----------------------------------

    st.subheader("📈 Driver Statistics")

    stats_df = pd.DataFrame({
        "Statistic": [
            "Experience",
            "Historical Finish",
            "Last 5 Finish",
            "Historical Rating"
        ],
        "Value": [
            latest["Driver_Experience"],
            round(latest["Driver_Historical_Avg_Finish"], 2),
            round(latest["Driver_Last5_Avg_Finish"], 2),
            round(latest["Driver_Historical_Avg_Rating"], 2)
        ]
    })

    left, right = st.columns(2)

    with left:

        st.metric(
            "Experience",
            int(latest["Driver_Experience"])
        )

        st.metric(
            "Historical Finish",
            round(
                latest["Driver_Historical_Avg_Finish"],
                2
            )
        )

    with right:

        st.metric(
            "Last 5 Finish",
            round(
                latest["Driver_Last5_Avg_Finish"],
                2
            )
        )

        st.metric(
            "Historical Rating",
            round(
                latest["Driver_Historical_Avg_Rating"],
                2
            )
        )

    # ----------------------------------
    # Team Performance
    # ----------------------------------

    st.markdown("---")
    st.subheader("🏆 Team Performance")

    t1, t2 = st.columns(2)

    with t1:
        st.metric(
            "Team Avg Finish",
            round(latest["Team_Historical_Avg_Finish"], 2)
        )

    with t2:
        st.metric(
            "Manufacturer Avg Finish",
            round(latest["Make_Historical_Avg_Finish"], 2)
        )

    # ----------------------------------
    # Performance Chart
    # ----------------------------------

    chart_df = pd.DataFrame({
        "Metric": [
            "Historical",
            "Last 5",
            "Prediction"
        ],
        "Finish": [
            latest["Driver_Historical_Avg_Finish"],
            latest["Driver_Last5_Avg_Finish"],
            prediction
        ]
    })

    st.subheader("📈 Performance Comparison")

    st.bar_chart(
        chart_df.set_index("Metric")
    )

    # ----------------------------------
    # Feature Importance
    # ----------------------------------

    feature_path = PROJECT_ROOT / "reports" / "feature_importance.png"

    if feature_path.exists():

        st.subheader("⭐ Feature Importance")

        st.image(str(feature_path), use_container_width=True)

    # ----------------------------------
    # Model Performance
    # ----------------------------------

    metrics_path = PROJECT_ROOT / "reports" / "metrics.txt"

    if metrics_path.exists():

        st.subheader("📋 Model Performance")

        with open(metrics_path) as f:
            st.text(f.read())



st.markdown("---")

st.markdown("""
<div style="text-align:center">

🏁 NASCAR AI Race Strategy Optimizer

Built using

Python • Streamlit • Scikit-Learn • XGBoost • CatBoost

© 2026

</div>
""", unsafe_allow_html=True)