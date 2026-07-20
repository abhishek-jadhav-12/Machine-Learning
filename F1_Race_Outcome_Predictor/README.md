# 🏎️ F1 Race Outcome Predictor

A Machine Learning-powered web application that predicts whether a Formula 1 driver is likely to finish on the podium based on race-related information. The application is built using **Python**, **Scikit-learn**, and **Streamlit**, providing an interactive dashboard for making real-time predictions.

---

🌐 Live Demo

🚀 Coming Soon...

---

## 📌 Project Overview

The goal of this project is to predict whether a Formula 1 driver will secure a podium finish using historical Formula 1 race data. The model is trained on race statistics, qualifying results, constructor information, and circuit details.

This project demonstrates the complete Machine Learning workflow including:

* Data Collection
* Data Preprocessing
* Feature Engineering
* Model Training
* Model Evaluation
* Model Deployment using Streamlit

---

## ✨ Features

* 🏎️ Predict Formula 1 podium finishes
* 📊 Interactive Streamlit web application
* ⚡ Real-time predictions
* 🤖 Machine Learning-based prediction model
* 📈 Prediction probability display
* 🎯 Easy-to-use interface

---

## 🛠️ Tech Stack

| Category             | Technologies  |
| -------------------- | ------------- |
| Programming Language | Python        |
| Machine Learning     | Scikit-learn  |
| Data Processing      | Pandas, NumPy |
| Model Serialization  | Joblib        |
| Visualization        | Matplotlib    |
| Web Framework        | Streamlit     |

---

## 📂 Project Structure

```text
F1_RACE_OUTCOME_PREDICTOR/
│
├── assets/
│   └── f1_logo.png
│
├── data/
│   ├── circuits.csv
│   ├── constructor_results.csv
│   ├── constructor_standings.csv
│   ├── constructors.csv
│   ├── driver_standings.csv
│   ├── drivers.csv
│   ├── f1_podium_dataset.csv
│   ├── lap_times.csv
│   ├── pit_stops.csv
│   ├── qualifying.csv
│   ├── races.csv
│   ├── results.csv
│   ├── seasons.csv
│   ├── sprint_results.csv
│   └── status.csv
│
├── images/
│
├── models/
│   ├── circuit_encoder.pkl
│   ├── constructor_encoder.pkl
│   └── podium_predictor.pkl
│
├── notebooks/
│   └── F1_Race_Outcome_Predictor.ipynb
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## 📊 Dataset

The project uses historical Formula 1 race data, including:

* Race Results
* Drivers
* Constructors
* Circuits
* Qualifying Results
* Pit Stops
* Lap Times
* Driver Standings
* Constructor Standings
* Sprint Results

The processed dataset used for model training is:

```text
data/f1_podium_dataset.csv
```

---

## 🤖 Machine Learning Model

The trained model predicts whether a driver will finish on the podium.

### Input Features

* Circuit
* Constructor
* Grid Position
* Starting Position

### Output

* 🟢 Podium Finish
* 🔴 Non-Podium Finish

Saved models are located in:

```text
models/
├── podium_predictor.pkl
├── circuit_encoder.pkl
└── constructor_encoder.pkl
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/abhishek-jadhav-12/F1_Race_Outcome_Predictor.git
```

### 2. Navigate to the project folder

```bash
cd F1_Race_Outcome_Predictor
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

---

## 📷 Application Preview

Add screenshots of your application inside the **images/** folder.

Example:

```text
images/
├── home_page.png
├── prediction_page.png
└── results.png
```

Then display them here using Markdown.

---

## 📈 Future Improvements

* Live Formula 1 API integration
* Driver performance statistics
* Weather-based race prediction
* Tire strategy analysis
* Model performance comparison
* Enhanced UI with Plotly visualizations
* Deploy using Streamlit Community Cloud

---

## 👨‍💻 Author

**Abhishek Shivprasad Jadhav**

GitHub: https://github.com/abhishek-jadhav-12 

LinkedIn: https://www.linkedin.com/in/abhishek-s-jadhav

---

## ⭐ Show Your Support

If you found this project useful, please consider giving it a ⭐ on GitHub. Your support is greatly appreciated!

---

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for more details.
