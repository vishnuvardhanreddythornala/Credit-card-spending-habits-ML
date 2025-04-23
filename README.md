# 💳 Credit Card Transaction Analysis Using Classification

This machine learning project analyzes credit card transactions to (1) classify expenses into predefined categories and (2) recommend credit card tier upgrades based on user spending behavior. Developed using a real-world Indian credit card dataset, the system integrates data preprocessing, model building, evaluation, and deployment via a Streamlit web app.

---

## 📌 Project Objectives

- 🔍 Predict the **expense category** of each transaction (e.g., Travel, Food, Bills).
- 📈 Recommend **credit card tier upgrades** for high-spending users.
- 🧠 Use **Random Forest Classifier** for robust, scalable performance.
- 🌐 Deploy predictions via a real-time, interactive **Streamlit** interface.

---

## 📊 Dataset Description

The dataset simulates real credit card usage in India, with 18,000+ records and 6 primary features:

| Feature     | Description                                      |
|-------------|--------------------------------------------------|
| City        | City where the transaction was made              |
| Date        | Date of the transaction                          |
| Card Type   | Type of card used (Silver, Gold, Platinum)       |
| Exp Type    | Category of expense (target variable 1)          |
| Gender      | Gender of the cardholder                         |
| Amount      | Transaction amount in INR                        |

Additional engineered features for upgrade prediction:
- **Avg_Amount**: User's average monthly spending
- **HighEnd_Freq**: Frequency of high-value expenses (Travel, Bills)

---

## 🧹 Data Preprocessing

- ✅ Label encoding for categorical features
- 📉 Outlier capping using IQR method (e.g., ₹200,000 cap)
- 🔄 Feature scaling (MinMax, StandardScaler as needed)
- 📊 Exploratory Data Analysis (EDA) using histograms, boxplots, heatmaps
- ⚙️ Engineered features to support upgrade prediction

---

## 🤖 ML Models Used

### 1. **Expense Category Prediction** (Multi-class classification)
- Algorithm: Random Forest
- Accuracy: **86%**
- F1-Score: **0.85**

### 2. **Card Tier Upgrade Recommendation** (Binary classification)
- Algorithm: Random Forest
- Accuracy: **89%**
- F1-Score: **0.89**

### Evaluation Metrics:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix

---

## 🗂️ Project Structure

├── app.py # Streamlit web app for predictions
├── Credit card spending habits.ipynb # Jupyter notebook for EDA + model training 
├── train_model.py # Model training and serialization script 
├── Credit card transactions - India - Simple.csv # Dataset 
├── rf_model.pkl # Trained model for expense category 
├── upgrade_model.pkl # Trained model for tier upgrades 
├── encoders.pkl # Encoders for transaction classification 
├── upgrade_encoders.pkl # Encoders for upgrade classification 
├── requirements.txt # Required Python packages
