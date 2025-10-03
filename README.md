TELCO CUSTOMER CHURN PREDICTION PROJECT

🎯 Project Goal

This project implements a Machine Learning solution to predict customer churn for a telecommunications company. The primary goal is to identify customers at high risk of leaving so that proactive retention efforts can be launched.

The core solution involves a Random Forest Classifier model trained on resampled data using SMOTE (Synthetic Minority Oversampling Technique) to account for severe class imbalance.

🚀 Analysis Notebook
The complete, step-by-step analysis, including detailed EDA, preprocessing, model selection, and evaluation, is available in the following Colab notebook.

➡️ Insert Colab Notebook Link Here: [[COLAB_LINK_HERE](https://drive.google.com/file/d/1DZmkfCJpLlAuwvMril73Ezh75mC4BuvD/view?usp=sharing)] ⬅️

🛠️ Key Steps and Findings
1. Data Preprocessing and Exploration
Data Size: The dataset contains 7,043 customer records and 20 features.

Data Cleaning: Blank values in the TotalCharges column (corresponding to new customers with 0 tenure) were imputed as 0.0 and the column was converted to a float data type.

Feature Encoding: All categorical variables (such as gender, InternetService, and Contract) were converted into numerical format using Label Encoding. These encoders are saved for production use (encoders.pkl).

Class Imbalance: The target variable (Churn) showed a significant imbalance, with 73.5% of customers not churning and 26.5% churning.

2. Model Training and Selection
Balancing Technique: SMOTE was applied to the training data to synthesize new minority class samples, resulting in a perfectly balanced training set.

Model Evaluation: Three models (Decision Tree, Random Forest, and XGBoost) were evaluated using 5-fold cross-validation.

Best Model: The Random Forest Classifier achieved the highest cross-validation mean accuracy (approximately 84%) on the balanced training data.

3. Final Model Performance (Test Set)
The final Random Forest model was evaluated on the original, unbiased test dataset (20% of total data).

Overall Accuracy: 77.86%

Churn Recall: 0.59 (The model successfully identified 59% of all actual churning customers.)

Churn Precision: 0.58 (Out of all customers predicted to churn, 58% were actually correct.)

Confusion Matrix Summary:

Correctly predicted Non-Churn (TN): 878

False Alarm (Predicted Churn, Actual No Churn) (FP): 158

Missed Churn (Predicted No Churn, Actual Churn) (FN): 154

Correctly predicted Churn (TP): 219

📂 Repository Artifacts
Customer_Churn_Prediction_using_ML (1).ipynb: The complete Jupyter/Colab notebook.

WA_Fn-UseC_-Telco-Customer-Churn.csv: The raw input data file.

customer_churn_model.pkl: The trained Random Forest model artifact.

encoders.pkl: The saved Label Encoders artifact for data preprocessing.

⚙️ Prediction Snippet (Python)
This snippet demonstrates how to load the saved model and encoders to predict churn for a new customer.

Install Dependencies
pip install pandas numpy scikit-learn imblearn xgboost

Python Code for Prediction
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# --- 1. Load Model and Encoders ---
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
    if column in input_data_df.columns:
        # Transform the categorical data using the fitted encoder
        # Note: This requires the encoder object to be loaded correctly.
        try:
            input_data_df[column] = encoder.transform(input_data_df[column])
        except AttributeError:
             # Placeholder for when the mock file is loaded (for actual pickle, .transform works)
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

🔗 Colab Link Placeholder

https://drive.google.com/file/d/1DZmkfCJpLlAuwvMril73Ezh75mC4BuvD/view?usp=sharing


