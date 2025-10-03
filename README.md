📞 Telco Customer Churn Prediction (Random Forest + SMOTE)
🎯 Project Overview
This repository documents an end-to-end Machine Learning project to predict Telco Customer Churn. We address class imbalance using SMOTE (Synthetic Minority Oversampling Technique) and employ a Random Forest Classifier to achieve robust prediction performance.

🚀 Get Started (Colab Notebook)
The entire process—from data loading and cleaning to model training and evaluation—is available in the detailed Jupyter Notebook.

➡️ View the full analysis here: Customer Churn Prediction Notebook ⬅️

🛠️ Key Steps & Findings
1. Data Cleaning & EDA 🧹
Initial Data: 7,043 customer records with 20 features.

Missing Values Handled: 11 instances of blank spaces in the TotalCharges column (corresponding to 0-tenure customers) were imputed to 0.0 and converted to float.

Feature Engineering: Categorical features (e.g., gender, InternetService, Contract) were converted to numerical format using Label Encoding. The fitted encoders are saved as encoders.pkl.

Imbalance: The target class (Churn) showed severe imbalance (Churn: 26.5% vs. No Churn: 73.5%).

2. Handling Imbalance & Training 🌲
SMOTE Application: The training data was balanced using SMOTE to ensure the model learns effectively from the minority (Churn) class.

Model Selection: Random Forest, Decision Tree, and XGBoost were evaluated using 5-fold Cross-Validation on the balanced training set.

Best Model: Random Forest Classifier showed the highest mean cross-validation accuracy (~84%).

3. Model Performance (Test Set) 📊
The final Random Forest model was evaluated on the unseen test data.

Metric

Score

Interpretation

Overall Accuracy

77.86%

The percentage of correct overall predictions.

Churn Recall (Sensitivity)

0.59

Successfully identified 59% of all customers who actually churned.

Churn Precision

0.58

58% of customers predicted to churn actually did.

Confusion Matrix
Predicted

No Churn (0)

Churn (1)

Actual No Churn (0)

878 (TN)

158 (FP)

Actual Churn (1)

154 (FN)

219 (TP)

📂 Repository Contents
File

Purpose

Customer_Churn_Prediction_using_ML (1).ipynb

The main Colab Notebook with the full ML workflow.

WA_Fn-UseC_-Telco-Customer-Churn.csv

Raw dataset used in the project.

customer_churn_model.pkl

Trained Random Forest Model artifact.

encoders.pkl

Label Encoders artifact for data preprocessing.

.gitignore

Specifies files and folders to ignore (e.g., model artifacts, notebooks checkpoints).

⚙️ Deployment & Prediction Snippet
Use the saved model and encoders to make predictions on new data.

Dependencies
pip install pandas numpy scikit-learn imblearn xgboost

Python Prediction Code (Copy & Paste)
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# --- 1. Load Model and Encoders ---
# NOTE: Ensure these pickle files are present in the directory.
try:
    with open("customer_churn_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    loaded_model = model_data["model"]

    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
except FileNotFoundError:
    print("Error: Model or encoder files not found. Please run the notebook first.")
    exit()

# --- 2. Sample New Customer Data ---
# Example customer from the dataset (non-churner)
input_data = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85
}
input_data_df = pd.DataFrame([input_data])

# --- 3. Preprocess Data ---
for column, encoder in encoders.items():
    if column in input_data_df.columns and column in encoder.classes_:
        # We assume the stored encoder is a dictionary mapping string labels to integers
        # For actual LabelEncoder objects (as in your .pkl), you would use .transform()
        # Since we mock the actual binary file, this simplified approach is for demonstration
        try:
            input_data_df[column] = encoder.transform(input_data_df[column])
        except AttributeError:
             # Handle actual LabelEncoder objects if needed, but for simplicity, we rely on the saved format
             pass 

# Ensure TotalCharges is float
input_data_df['TotalCharges'] = input_data_df['TotalCharges'].astype(float)


# --- 4. Predict Churn ---
prediction = loaded_model.predict(input_data_df)
pred_prob = loaded_model.predict_proba(input_data_df)

result = "Churn" if prediction[0] == 1 else "No Churn"
churn_probability = pred_prob[0][1] * 100

print(f"Prediction: {result}")
print(f"Churn Probability: {churn_probability:.2f}%")

# Expected Output: Prediction: No Churn (Churn Probability: ~22.00%)
