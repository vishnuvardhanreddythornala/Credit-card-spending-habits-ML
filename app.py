# app.py

import streamlit as st
import numpy as np
import pickle

# Load Expense Type Model
with open("rf_model.pkl", "rb") as f:
    model_exp = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    enc_exp = pickle.load(f)

# Load Upgrade Model
with open("upgrade_model.pkl", "rb") as f:
    model_upgrade = pickle.load(f)

with open("upgrade_encoders.pkl", "rb") as f:
    enc_upgrade = pickle.load(f)

st.set_page_config(page_title="Credit Card ML Suite")
tab1, tab2, tab3 = st.tabs(["🧾 Expense Category Predictor", "🏦 Card Upgrade Recommender", "🛍️ Merchant NLP Classifier"])


# ============================
# Tab 1: Expense Category Predictor
# ============================
with tab1:
    st.header("🧾 Predict Credit Card Expense Category")

    amount = st.number_input("💳 Transaction Amount (INR)", min_value=1, key="amount_1")
    card_type = st.selectbox("📇 Card Type", enc_exp["card"].classes_, key="card_1")
    city = st.selectbox("📍 City", enc_exp["city"].classes_, key="city_1")
    gender = st.radio("🧑‍🤝‍🧑 Gender", enc_exp["gender"].classes_, key="gender_1")

    if st.button("Predict Expense Type", key="predict_1"):
        try:
            card_enc = enc_exp["card"].transform([card_type])[0]
            city_enc = enc_exp["city"].transform([city])[0]
            gender_enc = enc_exp["gender"].transform([gender])[0]

            input_data = np.array([[city_enc, card_enc, gender_enc, amount]])
            pred = model_exp.predict(input_data)
            pred_label = enc_exp["exp"].inverse_transform(pred)[0]
            st.success(f"Predicted Expense Category: **{pred_label}**")
        except Exception as e:
            st.error(f"Error: {e}")

# ============================
# Tab 2: Card Tier Upgrade Predictor
# ============================
with tab2:
    st.header("🏦 Card Tier Upgrade Recommender")

    avg_amount = st.number_input("📊 Average Monthly Spending (INR)", min_value=100, key="amount_2")
    high_end_count = st.number_input("✈️ Travel/Bills Transactions per Month", min_value=0, step=1, key="highend_2")

    card_type = st.selectbox("📇 Current Card Type", enc_upgrade["card"].classes_, key="card_2")
    gender = st.selectbox("🧑‍🤝‍🧑 Gender", enc_upgrade["gender"].classes_, key="gender_2")
    city = st.selectbox("📍 City", enc_upgrade["city"].classes_, key="city_2")

    if st.button("Check Upgrade Eligibility", key="predict_2"):
        try:
            card_enc = enc_upgrade["card"].transform([card_type])[0]
            gender_enc = enc_upgrade["gender"].transform([gender])[0]
            city_enc = enc_upgrade["city"].transform([city])[0]

            input_data = np.array([[card_enc, gender_enc, city_enc, avg_amount, high_end_count]])
            pred = model_upgrade.predict(input_data)[0]

            if pred == 1:
                st.success("✅ Yes! Recommend a higher-tier card.")
            else:
                st.info("🔒 No upgrade needed at this time.")
        except Exception as e:
            st.error(f"Error: {e}")

# ============================
# Tab 3: NLP Merchant Classifier
# ============================
with tab3:
    st.header("🔍 Predict Expense Type from Merchant")
    merchant_input = st.text_input("Enter Merchant Name (e.g., Swiggy, IRCTC, Amazon)")

    if st.button("Predict Category"):
        try:
            with open("merchant_model.pkl", "rb") as f:
                merchant_model = pickle.load(f)
            with open("merchant_vectorizer.pkl", "rb") as f:
                merchant_vectorizer = pickle.load(f)
            with open("encoders.pkl", "rb") as f:
                encoders = pickle.load(f)
                le_exp = encoders["Expense Type"]

            vectorized_input = merchant_vectorizer.transform([merchant_input])
            predicted_category_code = merchant_model.predict(vectorized_input)[0]
            predicted_category = le_exp.inverse_transform([predicted_category_code])[0]

            st.success(f"Predicted Expense Category: **{predicted_category}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
