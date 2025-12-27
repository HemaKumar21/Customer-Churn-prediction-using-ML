📊 Customer Churn Prediction using Machine Learning
🔹 Project Overview

Customer churn is a critical problem for subscription-based businesses.
This project aims to predict whether a customer will churn or not using machine learning techniques on the Telco Customer Churn dataset.

The project covers data cleaning, exploratory data analysis (EDA), preprocessing, handling class imbalance, model training, evaluation, and deployment-ready prediction.

🔹 Dataset

Dataset: Telco Customer Churn

Rows: 7,043

Columns: 21

Target Variable: Churn (Yes / No)

Key Features:

Customer demographics (gender, senior citizen, dependents)

Account information (tenure, contract, payment method)

Service usage (internet service, streaming, security)

Billing details (monthly charges, total charges)

🔹 Technologies & Libraries Used

Python

Google Colab

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

Imbalanced-learn (SMOTE)

XGBoost

Pickle

🔹 Data Preprocessing

Removed unnecessary column (customerID)

Converted TotalCharges from object to numeric

Replaced missing values in TotalCharges

Label encoded categorical features

Encoded target variable (Churn: Yes → 1, No → 0)

Addressed class imbalance using SMOTE

🔹 Exploratory Data Analysis (EDA)

Distribution analysis using histograms and boxplots

Correlation heatmap for numerical features

Count plots for categorical features

Identified class imbalance in churn data

🔹 Model Training

Trained and compared multiple models using 5-fold cross-validation:

Model	CV Accuracy
Decision Tree	78%
Random Forest	84%
XGBoost	83%

✅ Random Forest performed best and was selected as the final model.

🔹 Model Evaluation

Test Set Performance (Random Forest):

Accuracy: ~78%

Precision (Churn): 0.58

Recall (Churn): 0.59

F1-score: 0.58

Confusion Matrix and Classification Report were used for evaluation.

🔹 Model Saving & Reuse

Trained model saved using pickle

Label encoders stored separately

Enables reuse without retraining

Files saved:

customer_churn_model.pkl

encoders.pkl

🔹 Prediction System

The project includes a prediction pipeline:

Accepts new customer input

Applies stored label encoders

Uses trained model to predict churn

Outputs Yes / No churn prediction

🔹 Project Structure
📁 Customer-Churn-Prediction
│── customer_churn_model.pkl
│── encoders.pkl
│── churn_prediction.ipynb
│── README.md

🔹 Results & Insights

Contract type and tenure strongly influence churn

Month-to-month customers have higher churn risk

SMOTE significantly improved model learning

Random Forest provided the best balance of accuracy and stability

🔹 Future Improvements

Hyperparameter tuning

Feature importance visualization

Deployment using Flask / FastAPI

Real-time prediction web app

🔹 Author

Hema Kumar V
AI & ML Enthusiast
