# ============================================================
# app.py — Streamlit Web App for Customer Churn Prediction
# Run with: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.markdown("*Predict whether a bank customer will leave using Machine Learning*")
st.markdown("---")

# ── Load & Train Model (cached so it runs only once) ─────────
@st.cache_resource
def load_and_train():
    df = pd.read_csv("data/Churn_Modelling.csv")
    df_model = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

    le = LabelEncoder()
    df_model['Gender']    = le.fit_transform(df_model['Gender'])
    df_model['Geography'] = le.fit_transform(df_model['Geography'])

    X = df_model.drop(columns=['Exited'])
    y = df_model['Exited']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=10,
                                    random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    cm  = confusion_matrix(y_test, model.predict(X_test))

    return model, scaler, df, acc, auc, cm

model, scaler, df, acc, auc, cm = load_and_train()

# ── Sidebar — Model Stats ─────────────────────────────────────
with st.sidebar:
    st.header("🤖 Model Info")
    st.metric("Algorithm",  "Random Forest")
    st.metric("Accuracy",   f"{acc*100:.2f}%")
    st.metric("AUC-ROC",    f"{auc:.4f}")
    st.metric("Dataset",    f"{len(df):,} customers")
    st.markdown("---")
    st.caption("Built with scikit-learn & Streamlit")

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📈 Data Analysis", "📋 Model Results"])

# ── TAB 1: Predict ────────────────────────────────────────────
with tab1:
    st.subheader("Enter Customer Details")
    st.markdown("Fill in the customer information and click **Predict** to see if they will churn.")

    col1, col2, col3 = st.columns(3)

    with col1:
        credit_score  = st.slider("Credit Score",       350, 850, 650)
        age           = st.slider("Age",                18,  92,  38)
        tenure        = st.slider("Tenure (years)",     0,   10,  5)

    with col2:
        geography     = st.selectbox("Country",         ["France", "Germany", "Spain"])
        gender        = st.selectbox("Gender",          ["Female", "Male"])
        num_products  = st.selectbox("Number of Products", [1, 2, 3, 4])

    with col3:
        balance          = st.number_input("Account Balance ($)", 0.0, 250000.0, 50000.0, step=1000.0)
        estimated_salary = st.number_input("Estimated Salary ($)", 0.0, 200000.0, 100000.0, step=1000.0)
        has_cr_card      = st.selectbox("Has Credit Card?",   ["Yes", "No"])
        is_active        = st.selectbox("Is Active Member?",  ["Yes", "No"])

    if st.button("🔮 Predict Churn", use_container_width=True, type="primary"):
        geo_map    = {"France": 0, "Germany": 1, "Spain": 2}
        gender_map = {"Female": 0, "Male": 1}
        yn_map     = {"Yes": 1, "No": 0}

        input_data = pd.DataFrame([{
            "CreditScore"    : credit_score,
            "Geography"      : geo_map[geography],
            "Gender"         : gender_map[gender],
            "Age"            : age,
            "Tenure"         : tenure,
            "Balance"        : balance,
            "NumOfProducts"  : num_products,
            "HasCrCard"      : yn_map[has_cr_card],
            "IsActiveMember" : yn_map[is_active],
            "EstimatedSalary": estimated_salary
        }])

        input_scaled = scaler.transform(input_data)
        prediction   = model.predict(input_scaled)[0]
        probability  = model.predict_proba(input_scaled)[0]

        st.markdown("---")
        r1, r2, r3 = st.columns(3)

        with r1:
            if prediction == 1:
                st.error("⚠️ This customer is likely to **CHURN**")
            else:
                st.success("✅ This customer is likely to **STAY**")

        with r2:
            st.metric("Churn Probability",  f"{probability[1]*100:.1f}%")

        with r3:
            st.metric("Stay Probability",   f"{probability[0]*100:.1f}%")

        # Risk level
        churn_pct = probability[1] * 100
        if churn_pct >= 70:
            st.warning("🔴 **High Risk** — Strongly recommend immediate retention offer")
        elif churn_pct >= 40:
            st.info("🟡 **Medium Risk** — Consider a loyalty incentive")
        else:
            st.success("🟢 **Low Risk** — Customer appears satisfied")


# ── TAB 2: Data Analysis ──────────────────────────────────────
with tab2:
    st.subheader("Exploratory Data Analysis")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Customers", f"{len(df):,}")
    m2.metric("Churned",         f"{df['Exited'].sum():,}")
    m3.metric("Churn Rate",      f"{df['Exited'].mean()*100:.1f}%")
    m4.metric("Avg Age",         f"{df['Age'].mean():.0f} yrs")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(5, 4))
        df['Exited'].value_counts().plot.pie(
            labels=['Stayed', 'Churned'], autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c'], ax=ax, startangle=90
        )
        ax.set_title("Churn Distribution")
        ax.set_ylabel("")
        st.pyplot(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(5, 4))
        geo = df.groupby('Geography')['Exited'].mean().mul(100)
        geo.plot(kind='bar', color=['#3498db','#e67e22','#9b59b6'], ax=ax, rot=0)
        ax.set_title("Churn Rate by Country")
        ax.set_ylabel("Churn Rate (%)")
        st.pyplot(fig)

    c3, c4 = st.columns(2)

    with c3:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(df[df['Exited']==0]['Age'], bins=25, alpha=0.7,
                color='#2ecc71', label='Stayed')
        ax.hist(df[df['Exited']==1]['Age'], bins=25, alpha=0.7,
                color='#e74c3c', label='Churned')
        ax.set_title("Age Distribution vs Churn")
        ax.set_xlabel("Age")
        ax.legend()
        st.pyplot(fig)

    with c4:
        fig, ax = plt.subplots(figsize=(5, 4))
        active = df.groupby('IsActiveMember')['Exited'].mean().mul(100)
        active.index = ['Inactive', 'Active']
        active.plot(kind='bar', color=['#e74c3c','#2ecc71'], ax=ax, rot=0)
        ax.set_title("Churn Rate: Active vs Inactive")
        ax.set_ylabel("Churn Rate (%)")
        st.pyplot(fig)


# ── TAB 3: Model Results ──────────────────────────────────────
with tab3:
    st.subheader("Model Evaluation")

    e1, e2, e3 = st.columns(3)
    e1.metric("Accuracy",  f"{acc*100:.2f}%")
    e2.metric("AUC-ROC",   f"{auc:.4f}")
    e3.metric("Model",     "Random Forest")

    st.markdown("---")
    mc1, mc2 = st.columns(2)

    with mc1:
        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Stayed', 'Churned'],
                    yticklabels=['Stayed', 'Churned'])
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        st.pyplot(fig)

    with mc2:
        st.markdown("**Feature Importances**")
        feature_names = ['CreditScore','Geography','Gender','Age','Tenure',
                         'Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
        fig, ax = plt.subplots(figsize=(5, 4))
        importances.plot(kind='barh', color='#3498db', ax=ax)
        ax.set_title("Feature Importances")
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("**Dataset Sample**")
    st.dataframe(df.head(10), use_container_width=True)
