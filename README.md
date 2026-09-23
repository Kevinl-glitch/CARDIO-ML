# CardioML — Cardiovascular Disease Risk Assessment System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask%203.0+-green.svg)](https://palletsprojects.com/p/flask/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn%201.6+-orange.svg)](https://scikit-learn.org/)
[![Deploy on Vercel](https://img.shields.io/badge/Deploy-Vercel-black.svg)](https://vercel.com/)
[![License](https://img.shields.io/badge/Course-Darshan%20University%20MLDL%20SOP-red.svg)]()

An end-to-end Machine Learning web application designed and built in accordance with the **Darshan University Computer Engineering Department Standard Operating Procedure (SOP)** for Student Project Execution (Weeks 1 through 7).

---

## 📌 Project Overview

**CardioML** analyzes clinical and physiological health indicators to estimate individual cardiovascular disease (CVD) risk. The application provides an early warning mechanism using a tuned **Gradient Boosting Classifier** evaluated on 70,000 patient records.

### Key Performance Benchmarks:
- **Test Accuracy:** `73.04%` (SOP Target: 73.1% - 73.4%)
- **ROC-AUC Score:** `79.88%` (SOP Target: 79.7% - 80.1%)
- **F1 Score:** `71.53%` (SOP Target: 71.7%)
- **Cross-Validation:** 5-Fold Stratified K-Fold
- **Top Predictors:** Systolic Blood Pressure (`ap_hi`: 70.4%), Age (`age_years`: 13.5%), Cholesterol (`cholesterol`: 7.4%), Body Mass Index (`bmi`: 2.8%).

---

## 📅 Weekly Deliverables (Weeks 1 – 7)

| Week | Milestone | Deliverable |
| :---: | :--- | :--- |
| **Week 1** | Problem Definition and Dataset Exploration | [`Week1_Cardio_Problem_Definition_EDA.ipynb`](./Week1_Cardio_Problem_Definition_EDA.ipynb) |
| **Week 2** | Data Cleaning, Preprocessing & Feature Engineering | [`Week2_Cardio_Data_Cleaning_Preprocessing.ipynb`](./Week2_Cardio_Data_Cleaning_Preprocessing.ipynb) |
| **Week 3** | Model Creation (Library + Scratch Algorithm) | [`Week3_Cardio_Model_Creation.ipynb`](./Week3_Cardio_Model_Creation.ipynb) |
| **Week 4** | Model Evaluation, ROC & Overfitting Diagnostics | [`Week4_Cardio_Model_Evaluation.ipynb`](./Week4_Cardio_Model_Evaluation.ipynb) |
| **Week 5** | Advanced Models Comparison & Hyperparameter Tuning | [`Week5_Cardio_Advanced_Model_Training.ipynb`](./Week5_Cardio_Advanced_Model_Training.ipynb) |
| **Week 6** | Frontend Interface Architecture & Dynamic Logic | [`Week6_Cardio_Frontend_Design.ipynb`](./Week6_Cardio_Frontend_Design.ipynb) & `templates/` |
| **Week 7** | Flask Backend Serving & Cloud Deployment | [`Week7_Cardio_Backend_and_Deployment.ipynb`](./Week7_Cardio_Backend_and_Deployment.ipynb) & `app.py` |

---

## 🚀 Live Web Interface (SOP Pages 7, 8, 9)

- **Landing Page (`/` - SOP Page 7):** Hero section with *Early Warning System* badge, 4 capability cards (*73.4% Accuracy*, *Instant Results*, *Secure & Private*, *Confidence Scores*).
- **CVD Prediction Form (`/predict` - SOP Page 8):** Dynamic client-side calculation of **BMI** and **Pulse Pressure** (`ap_hi - ap_lo`), real-time risk classification, probability confidence meter, and personal risk factor detection.
- **Model Details (`/model` - SOP Page 8 bottom):** Complete transparency dashboard displaying model hyperparameters, performance metrics, and interactive feature importance bars.
- **Data Insights (`/insights` - SOP Page 9):** Data cleaning funnel (70,000 raw → 68,573 cleaned), clinical understanding of CVD, and healthy target ranges table.
- **REST API (`/api/predict`, `/health`):** JSON endpoints for programmatic inference and uptime monitoring.

---

## 💻 Local Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/cardio-ml-project.git
cd cardio-ml-project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Flask Web App
```bash
python app.py
```
Open **`http://127.0.0.1:5000`** in your browser.

---

## ☁️ Deployment Guide

### Deploy to Vercel (Zero Configuration)
This repository is configured with `vercel.json` and `api/index.py` for direct deployment on Vercel:

#### Method 1: Via Vercel Web Dashboard (Easiest)
1. Push this repository to **GitHub**.
2. Go to [Vercel](https://vercel.com) and log in.
3. Click **Add New...** -> **Project**.
4. Import your GitHub repository `cardio-ml-project`.
5. Keep default settings and click **Deploy**.
6. Your app will be live on `https://<your-project>.vercel.app` in ~1 minute!

#### Method 2: Via Vercel CLI
```bash
npx vercel
```
Follow the interactive prompts to deploy directly from your terminal.

---

### Deploy to Render.com (Alternative Free Hosting)
1. Push your repository to GitHub.
2. Log into [Render.com](https://render.com) and click **New + Web Service**.
3. Connect your repository.
4. Set **Build Command:** `pip install -r requirements.txt`
5. Set **Start Command:** `gunicorn app:app`
6. Click **Create Web Service**.

*For PythonAnywhere and Railway instructions, refer to [`DEPLOYMENT_GUIDE.md`](./DEPLOYMENT_GUIDE.md).*

---

## 🏛️ Academic Compliance
- **Institution:** Darshan University
- **Department:** Computer Engineering Department
- **Course:** Machine Learning & Deep Learning Project (MLDL SOP)
- **Academic Year:** 2025–26
