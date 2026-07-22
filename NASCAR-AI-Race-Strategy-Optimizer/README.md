# 🏁 NASCAR AI Race Strategy Optimizer

<div align="center">

### Machine Learning Powered Race Finish Prediction & AI Strategy Recommendation System

Predict NASCAR race finishing positions using Machine Learning and receive AI-powered race strategy recommendations through an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-Regression-green)
![CatBoost](https://img.shields.io/badge/CatBoost-AI-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

</div>

---

# 🌐 Live Demo

🚀 **Live Application**

> **Coming Soon...**

The application will be deployed on **Streamlit Community Cloud**.

---

# 📌 Project Overview

The **NASCAR AI Race Strategy Optimizer** is an end-to-end Machine Learning project that predicts a NASCAR driver's finishing position based on historical race data, driver performance, team statistics, track characteristics, and starting position.

In addition to prediction, the application generates AI-powered race strategy recommendations to help optimize race performance.

The project demonstrates a complete Machine Learning workflow including:

- Data Collection
- Data Cleaning
- Feature Engineering
- Data Preprocessing
- Model Training
- Model Evaluation
- Prediction Pipeline
- AI Strategy Recommendation
- Streamlit Deployment

---

# 🚀 Features

- 🏁 Predict NASCAR race finishing position
- 📊 AI-powered race strategy recommendations
- 👤 Driver performance analysis
- 🏆 Team performance insights
- 📈 Historical performance comparison
- ⭐ Feature importance visualization
- 📋 Model performance evaluation
- 🎨 Professional NASCAR-themed Streamlit dashboard
- ⚡ Interactive prediction interface

---

# 🛠️ Tech Stack

### Programming Language

- Python

### Machine Learning

- Scikit-Learn
- XGBoost
- CatBoost

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Streamlit

### Model Persistence

- Joblib

### Development Tools

- VS Code
- Git
- GitHub

---

# 📂 Project Structure

```text
NASCAR-AI-Race-Strategy-Optimizer/
│
├── .github/
│   └── workflows/
│       └── python.yml
│
├── assets/
│   ├── banner.png
│   ├── logo.png
│   ├── dashboard.png          # Add after deployment
│   ├── prediction.png         # Add after deployment
│   └── strategy.png           # Add after deployment
│
├── data/
│   ├── raw/
│   │   ├── cup_series.parquet
│   │   ├── nxs_series.parquet
│   │   └── truck_series.parquet
│   │
│   └── processed/
│       ├── merged_dataset.parquet
│       ├── cleaned_dataset.parquet
│       └── feature_engineered.parquet
│
├── models/
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   ├── linear_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── xgboost.pkl
│   └── catboost.pkl
│
├── notebooks/
│   ├── data_exploration.ipynb
│   └── feature_engineering.ipynb
│
├── reports/
│   ├── actual_vs_predicted.png
│   ├── error_distribution.png
│   ├── feature_importance.csv
│   ├── feature_importance.png
│   ├── metrics.txt
│   ├── model_comparison.csv
│   └── residual_plot.png
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── strategy_engine.py
│   ├── explain.py
│   └── utils.py
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

# 📊 Machine Learning Pipeline

## 1. Data Collection

- NASCAR historical race dataset
- Multiple racing seasons
- Driver statistics
- Team information
- Track characteristics

---

## 2. Data Cleaning

- Missing value handling
- Duplicate removal
- Data validation
- Dataset preprocessing

---

## 3. Feature Engineering

Created several predictive features including:

- Driver Experience
- Driver Historical Average Finish
- Driver Last 5 Average Finish
- Driver Historical Rating
- Team Historical Average Finish
- Manufacturer Historical Average Finish
- Track Historical Average Finish
- Track Type
- Start Category

---

## 4. Data Preprocessing

- Train-Test Split
- Missing Value Imputation
- One-Hot Encoding
- Column Transformer
- Machine Learning Pipeline

---

## 5. Model Training

The following regression models were trained and compared:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- CatBoost Regressor

The best-performing model is automatically selected and saved for prediction.

---

## 6. Prediction

The trained model predicts:

- Expected finishing position
- Driver performance level
- Race competitiveness

---

## 7. AI Strategy Engine

Based on the predicted outcome, the application provides:

- 🛞 Tire Strategy
- ⛽ Fuel Strategy
- 🏁 Pit Stop Recommendation
- 🚦 Starting Strategy
- 📈 Performance Recommendation

---

# 📈 Model Evaluation

The models were evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

Additional evaluation reports include:

- Feature Importance
- Actual vs Predicted Plot
- Residual Plot
- Error Distribution
- Model Comparison

---

# 🖥️ Dashboard

The Streamlit dashboard provides:

- Driver Selection
- Team Selection
- Manufacturer Selection
- Track Selection
- Series Selection
- Starting Position Selection

It displays:

- 🏁 Predicted Finish Position
- ⭐ Driver Rating
- 🎯 Prediction Confidence
- 📊 AI Strategy Recommendations
- 👤 Driver Statistics
- 🏆 Team Performance
- 📈 Performance Comparison Charts
- ⭐ Feature Importance
- 📋 Model Evaluation Metrics

---

# 📷 Screenshots

### 🏁 Dashboard

> *(Screenshot will be added after deployment.)*

---

### 📊 Prediction

> *(Screenshot will be added after deployment.)*

---

### 🧠 AI Strategy

> *(Screenshot will be added after deployment.)*

---

### ⭐ Feature Importance

> *(Screenshot will be added after deployment.)*

---

# ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/abhishek-jadhav-12/NASCAR-AI-Race-Strategy-Optimizer.git
```

### Navigate to the project directory

```bash
cd NASCAR-AI-Race-Strategy-Optimizer
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

# 🎯 Future Improvements

- 🌦️ Weather-based race prediction
- ⛽ Fuel consumption optimization
- 🛞 Tire degradation prediction
- 🏎️ Driver comparison dashboard
- 📡 Live race telemetry integration
- 📈 Interactive Plotly visualizations
- 🤖 Deep Learning-based prediction models
- ☁️ Cloud deployment enhancements

---

# 👨‍💻 Author

## Abhishek Jadhav

**B.Tech Computer Science & Engineering**

Machine Learning Enthusiast | Python Developer | Data Science Enthusiast

**Skills**

- Python
- Machine Learning
- Data Analysis
- Streamlit
- Scikit-Learn
- XGBoost
- CatBoost
- Git & GitHub

---

# ⭐ Support

If you found this project useful, consider giving it a **⭐ Star** on GitHub.

Feedback, suggestions, and contributions are always welcome!

---

<div align="center">

### 🏁 Thank you for visiting this project!

Made with ❤️ using Python, Machine Learning, and Streamlit.

</div>