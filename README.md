# SmartLoan AI – Intelligent Loan Default Prediction & Risk Assessment Platform

## Overview

SmartLoan AI is a machine learning-powered web application designed to predict the likelihood of loan default and assist financial institutions in making data-driven lending decisions.

The platform analyzes applicant information such as income, credit score, employment status, work experience, loan amount, and existing liabilities to generate a risk score, default probability, and loan approval recommendation.

Built using Python, Flask, Machine Learning, HTML, CSS, and SQLite, SmartLoan AI demonstrates the practical application of Artificial Intelligence in the banking and financial services sector.

---

## Key Features

### AI-Powered Loan Risk Assessment

* Predicts probability of loan default.
* Generates risk scores from 0–100.
* Classifies applicants into:

  * Low Risk
  * Medium Risk
  * High Risk

### Intelligent Loan Recommendation

Provides automated recommendations:

* Approve
* Approve with Conditions
* Reject

### Explainable AI (XAI)

Displays the factors influencing each prediction:

* Credit Score Analysis
* Debt-to-Income Ratio
* Employment Status
* Work Experience
* Existing Loans
* Age-Based Risk Assessment

### User Authentication

* Secure Registration
* Secure Login
* Session Management
* Password Hashing

### Analytics Dashboard

* Total Applications
* Approved Loans
* Rejected Loans
* Average Risk Score
* Risk Distribution Analytics
* Recent Predictions

### Prediction History

Stores all previous predictions for future analysis and reporting.

---

## Technology Stack

### Frontend

* HTML5
* CSS3
* JavaScript

### Backend

* Python
* Flask

### Machine Learning

* Scikit-Learn
* Random Forest Classifier
* Pandas
* NumPy

### Database

* SQLite

### Model Serialization

* Joblib

---

## Machine Learning Workflow

### Input Features

The model evaluates:

* Age
* Annual Income
* Loan Amount
* Credit Score
* Employment Type
* Years of Experience
* Existing Loans
* Marital Status

### Prediction Process

Applicant Data
↓
Data Preprocessing
↓
Random Forest Model
↓
Default Probability Prediction
↓
Risk Score Generation
↓
Recommendation Engine
↓
Explainable AI Analysis

---

## Risk Classification

| Risk Score | Risk Level  | Recommendation          |
| ---------- | ----------- | ----------------------- |
| 0–34       | Low Risk    | Approve                 |
| 35–64      | Medium Risk | Approve with Conditions |
| 65–100     | High Risk   | Reject                  |

---

## Project Structure

```text
SmartLoan-AI/
│
├── app.py
├── database.py
├── model_explainer.py
├── train_model.py
├── requirements.txt
│
├── model_pipeline.joblib
├── model_metadata.joblib
├── smartloan.db
│
├── templates/
│   ├── home.html
│   ├── login.html
│   ├── predict.html
│   ├── results.html
│   ├── dashboard.html
│   ├── history.html
│   └── about.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
└── synthetic_loan_data.csv
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/smartloan-ai.git
cd smartloan-ai
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Application will start at:

```text
http://127.0.0.1:5000
```

---

## Model Performance

Model Used:

* Random Forest Classifier

Performance Metrics:

* Accuracy: Stored in model_metadata.joblib
* Feature Importance Analysis
* Explainable AI Support

---

## Future Enhancements

* Real Bank Dataset Integration
* Email Notifications
* PDF Loan Reports
* Credit Bureau API Integration
* Advanced Risk Visualization
* Deep Learning Models
* Cloud Deployment
* Role-Based Access Control

---

## Use Cases

* Banks
* NBFCs
* Fintech Companies
* Credit Risk Departments
* Lending Platforms
* Financial Analytics Systems

---

## Learning Outcomes

This project demonstrates practical experience in:

* Machine Learning
* Predictive Analytics
* Flask Development
* Database Management
* Authentication Systems
* Explainable AI
* Data Visualization
* Financial Risk Assessment

---

## Author

Arpreet K

Capstone Project – SmartLoan AI

AI-Powered Loan Default Prediction & Risk Assessment Platform
