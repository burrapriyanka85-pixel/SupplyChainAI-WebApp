# SupplyChainAI-WebApp

A Streamlit-based web application for Supply Chain Analytics and Demand Forecasting using machine learning models.

---

## 🚀 Overview

SupplyChainAI-WebApp is an interactive machine learning–powered dashboard that helps analyze supply chain performance, visualize operational KPIs, and predict demand using real-world datasets.  
This project is built using **Streamlit**, **Python**, and **Scikit-learn**, and is fully deployable on **Streamlit Cloud**.

---

## ✨ Features

- 📊 **Interactive Dashboard**
  - View supply chain metrics and KPIs
  - Explore demand and inventory insights

- 🤖 **ML Model Integration**
  - Machine learning model (Random Forest / Regression)
  - Predict demand or lead time using uploaded data

- 📁 **File Upload Support**
  - Upload CSV data and visualize instantly
  - Automatic preprocessing and analysis

- 📈 **Data Visualizations**
  - Charts using Matplotlib & Seaborn
  - Trend analysis and summary insights

- 🌐 **Streamlit Cloud Deployment**
  - Runs directly from GitHub
  - No installation required by the user

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **Pandas, NumPy**
- **Scikit-Learn**
- **Matplotlib, Seaborn**
- **Joblib** (model loading/saving)

---

## 📦 Installation (Local)
pip install -r requirements.txt
streamlit run app.py

📁 Project Structure
SupplyChainAI-WebApp/
│── app.py
│── requirements.txt
│── README.md
│── model.pkl / .joblib  (optional)
│── dataset.csv          (optional)
└── .streamlit/
    └── config.toml      (optional)

```bash
