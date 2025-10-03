🎯 Telco Customer Churn Prediction using Machine Learning

💡 Project Goal
The primary objective of this project is to build a machine learning model capable of predicting customer churn for a telecommunications company. By accurately identifying customers likely to leave, the company can proactively implement retention strategies.

🚀 Quick Start: Run the Notebook!
The entire analysis, preprocessing, training, and evaluation process is documented and runnable directly in the provided Jupyter/Colab notebook.

Step

Description

Link 

View Notebook

Explore the step-by-step data analysis and modeling process.

Customer Churn Prediction Notebook

Data Source

Original dataset used for training and evaluation.

WA_Fn-UseC_-Telco-Customer-Churn.csv

📊 Key Results & Model Performance
The final model used for prediction is a Random Forest Classifier, trained after applying SMOTE to handle the severe class imbalance in the training data.

Metric

Score (Test Set)

📝 Interpretation

Accuracy

~77.8%

Overall correct predictions.

Precision (Churn)

0.58

Out of all customers predicted to churn, 58% actually did.

Recall (Churn)

0.59

The model correctly identified 59% of all actual churning customers.

Confusion Matrix (Test Set)
Predicted

No Churn (0)

Churn (1)

Actual No Churn (0)

878 (True Negative)

158 (False Positive)

Actual Churn (1)

154 (False Negative)

219 (True Positive)

🛠️ Repository Structure
File/Directory

Description

README.md

(You are here) Project overview and guide.

Customer_Churn_Prediction_using_ML (1).ipynb

The Jupyter Notebook containing all code and analysis.

WA_Fn-UseC_-Telco-Customer-Churn.csv

The raw customer data used for the project.

customer_churn_model.pkl

Trained Model (Random Forest Classifier) ready for deployment.

encoders.pkl

Label Encoders necessary to preprocess new data for prediction.

.gitignore

Standard file to exclude unnecessary files from version control.

💻 Setup & Dependencies (Copy/Paste)
To set up the environment and run the notebook locally, you need the following libraries.

# Clone the repository
git clone [https://github.com/YourUsername/Telco-Churn-Prediction.git](https://github.com/YourUsername/Telco-Churn-Prediction.git)
cd Telco-Churn-Prediction


# Install the necessary Python packages
pip install pandas numpy scikit-learn matplotlib seaborn imblearn xgboost

🐍 Python Code Snippet for Prediction
You can use the saved model and encoders to predict churn for a new customer by transforming their categorical features, like this:

import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# --- 1. Load Model and Encoders ---
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)
loaded_model = model_data["model"]

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

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
    # Only try to transform if the column exists in the encoder keys
    if column in input_data_df.columns:
        # Note: If a value is unseen during fit, this line will raise an error.
        # Ensure all possible categories are present during training setup.
        input_data_df[column] = encoder.transform(input_data_df[column])

# --- 4. Predict Churn ---
prediction = loaded_model.predict(input_data_df)
pred_prob = loaded_model.predict_proba(input_data_df)

result = "Churn" if prediction[0] == 1 else "No Churn"
churn_probability = pred_prob[0][1] * 100

print(f"Prediction: {result}")
print(f"Churn Probability: {churn_probability:.2f}%")

# Expected Output (based on notebook):
# Prediction: No Churn
# Churn Probability: 22.00%
customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn
7590-VHVEG,Female,0,Yes,No,1,No,No phone service,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85,No
5575-GNVDE,Male,0,No,No,34,Yes,No,DSL,Yes,No,Yes,No,No,No,One year,No,Mailed check,56.95,1889.5,No
3668-QPYBK,Male,0,No,No,2,Yes,No,DSL,Yes,Yes,No,No,No,No,Month-to-month,Yes,Mailed check,53.85,108.15,Yes
7795-CFOCW,Male,0,No,No,45,No,No phone service,DSL,Yes,No,Yes,Yes,No,No,One year,No,Bank transfer (automatic),42.3,1840.75,No
9237-HQITU,Female,0,No,No,2,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,70.7,151.65,Yes

... (File contains 7043 rows)
# This is a placeholder file. In a real scenario, this would be a binary
# pickle file containing the trained RandomForestClassifier model object.
# To regenerate the actual file, you must run the Jupyter notebook.
#
# model_data = {"model": rfc, "features_names": X.columns.tolist()}
# with open("customer_churn_model.pkl", "wb") as f:
#   pickle.dump(model_data, f)
#
# Contents: A mock representation of the pickled data structure.
model_data = {
    "model": "RandomForestClassifier(random_state=42) - FITTED",
    "features_names": [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
    ]
}
# This is a placeholder file. In a real scenario, this would be a binary
# pickle file containing the dictionary of fitted LabelEncoder objects.
# The code snippet in the README shows how this file is loaded and used.
#
# Contents: A mock representation of the pickled dictionary of encoders.
encoders = {
    'gender': {'Female': 0, 'Male': 1},
    'Partner': {'No': 0, 'Yes': 1},
    'Dependents': {'No': 0, 'Yes': 1},
    'PhoneService': {'No': 0, 'Yes': 1},
    'MultipleLines': {'No': 0, 'No phone service': 1, 'Yes': 2},
    'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
    'OnlineSecurity': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'OnlineBackup': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'DeviceProtection': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'TechSupport': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'StreamingTV': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'StreamingMovies': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
    'PaperlessBilling': {'No': 0, 'Yes': 1},
    'PaymentMethod': {'Bank transfer (automatic)': 0, 'Credit card (automatic)': 1, 'Electronic check': 2, 'Mailed check': 3}
}


colab 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]((https://drive.google.com/file/d/1DZmkfCJpLlAuwvMril73Ezh75mC4BuvD/view?usp=sharing))
