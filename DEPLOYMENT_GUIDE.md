# CardioML — Deployment & Execution Guide (Week 7 Deliverable)

This document provides step-by-step instructions to run the **CardioML** application locally and deploy it to popular **free cloud hosting platforms** in compliance with the **Darshan University MLDL SOP Project guidelines (Week 7: Create backend, Deployment)**.

---

## 1. Running Locally on Your Machine

### Prerequisites
- Python 3.10+ (or Anaconda environment)
- Dependencies installed:
  ```bash
  pip install -r requirements.txt
  ```

### Step 1: Retrain / Export Artifacts (Optional)
If you wish to retrain the model and regenerate `model.pkl`, `scaler.pkl`, and `model_metadata.json`:
```bash
python train_and_export.py
```

### Step 2: Start the Flask Application
Run the main web server:
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 2. Free Cloud Hosting Deployment Options

### Option A: Render.com (Recommended Free Hosting)
Render offers a free tier for Python web services with automated GitHub deployment and free SSL.

1. **Push your code to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "CardioML complete project through Week 7"
   git branch -M main
   git remote add origin https://github.com/<your-username>/cardio-ml-project.git
   git push -u origin main
   ```
2. **Log in to [Render.com](https://render.com/):**
   - Click **New +** and select **Web Service**.
   - Connect your GitHub repository.
3. **Configure Settings:**
   - **Name:** `cardioml-prediction-app` (or any unique name)
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Instance Type:** `Free`
4. **Deploy:**
   - Click **Create Web Service**.
   - Within 2–3 minutes, Render will build and deploy your application to a live public URL:
     `https://cardioml-prediction-app.onrender.com`

---

### Option B: PythonAnywhere (Free Python Hosting)
1. Register for a free account at [pythonanywhere.com](https://www.pythonanywhere.com).
2. Go to the **Files** tab and upload your project zip or clone via Git in the Bash console:
   ```bash
   git clone https://github.com/<your-username>/cardio-ml-project.git
   ```
3. Open a **Bash Console** and create a virtual environment:
   ```bash
   mkvirtualenv --python=/usr/bin/python3.10 cardio-env
   pip install -r requirements.txt
   ```
4. Go to the **Web** tab:
   - Click **Add a new web app** (Manual configuration, Python 3.10).
   - In **Code** section, set **Source code** to `/home/<username>/cardio-ml-project`.
   - Set **Working directory** to `/home/<username>/cardio-ml-project`.
   - Set **Virtualenv** to `/home/<username>/.virtualenvs/cardio-env`.
   - Edit the **WSGI configuration file**:
     ```python
     import sys
     path = '/home/<username>/cardio-ml-project'
     if path not in sys.path:
         sys.path.append(path)

     from app import app as application
     ```
5. Click **Reload** to make your site live at:
   `https://<username>.pythonanywhere.com`

---

### Option C: Railway.app (Free / Hobby Container Hosting)
1. Create a free account at [Railway.app](https://railway.app).
2. Click **New Project** -> **Deploy from GitHub repo**.
3. Select your repository. Railway automatically detects `Dockerfile` or `Procfile` and launches the application.

---

## 3. REST API Documentation (Week 7 Deliverable)

### 1. Model Prediction Endpoint
- **URL:** `/api/predict`
- **Method:** `POST`
- **Header:** `Content-Type: application/json`
- **Request Body Example:**
  ```json
  {
    "age_years": 50,
    "gender": 2,
    "height": 175,
    "weight": 70.5,
    "ap_hi": 120,
    "ap_lo": 80,
    "cholesterol": 1,
    "gluc": 1,
    "smoke": 0,
    "alco": 0,
    "active": 1
  }
  ```
- **Response Example:**
  ```json
  {
    "status": "success",
    "prediction": 0,
    "probability": 23.4,
    "risk_level": "Low Risk",
    "bmi": 23.02,
    "pulse_pressure": 40,
    "risk_factors": []
  }
  ```

### 2. Model Metadata & Performance Endpoint
- **URL:** `/api/model-info`
- **Method:** `GET`
- **Response:** JSON object containing hyperparameters, 5-fold CV metrics, test accuracy, F1-score, and sorted feature importances.

### 3. Health Check Endpoint
- **URL:** `/health`
- **Method:** `GET`
- **Response:** `{"status": "healthy", "service": "CardioML", "version": "1.0.0"}`

---

## 4. Verification & Testing Checklist

- [x] All 7 Weekly Notebooks are present and validated (`Week1` through `Week7`).
- [x] Gradient Boosting Classifier tuned and evaluated on held-out test data (Accuracy 73.04%, ROC-AUC 79.88%).
- [x] Flask backend connects to all datasets and model artifacts without unpickling warnings.
- [x] Frontend faithfully implements pages 7, 8, and 9 of Darshan University MLDL SOP.
- [x] Live dynamic calculation of BMI and pulse pressure works on the client.
- [x] Zero-config deployment configurations created (`Procfile`, `render.yaml`, `Dockerfile`, `requirements.txt`).
