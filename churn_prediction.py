# ============================================================
# CUSTOMER CHURN PREDICTION PROJECT
# Dataset : Churn_Modelling.csv (10,000 Bank Customers)
# Target  : Exited → 1 = Churned, 0 = Stayed
# Models  : Logistic Regression + Random Forest
# ============================================================

# ── STEP 1: Import Libraries ─────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)

os.makedirs("outputs", exist_ok=True)

print("=" * 58)
print("   CUSTOMER CHURN PREDICTION — BANK ML PROJECT")
print("=" * 58)


# ── STEP 2: Load Dataset ──────────────────────────────────────
print("\n📂 STEP 1: Loading Dataset...")
df = pd.read_csv("data/Churn_Modelling.csv")

print(f"   ✅ Rows    : {df.shape[0]:,}")
print(f"   ✅ Columns : {df.shape[1]}")
print(f"\n   Preview:")
print(df.head(3).to_string())


# ── STEP 3: Explore Data ──────────────────────────────────────
print("\n" + "─" * 58)
print("📊 STEP 2: Exploring the Data...")

stayed  = df['Exited'].value_counts()[0]
churned = df['Exited'].value_counts()[1]

print(f"\n   Total Customers  : {len(df):,}")
print(f"   Stayed  (0)      : {stayed:,}  ({stayed/len(df)*100:.1f}%)")
print(f"   Churned (1)      : {churned:,}  ({churned/len(df)*100:.1f}%)")
print(f"   Missing Values   : {df.isnull().sum().sum()}")

print(f"\n   Numeric Summary:")
print(df[['CreditScore','Age','Tenure','Balance','EstimatedSalary']].describe().round(2).to_string())


# ── STEP 4: Visualizations ────────────────────────────────────
print("\n" + "─" * 58)
print("📈 STEP 3: Creating EDA Visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle("Bank Customer Churn — Exploratory Data Analysis",
             fontsize=16, fontweight='bold', y=1.01)

# 1. Churn distribution pie
counts = df['Exited'].value_counts()
axes[0,0].pie(counts, labels=['Stayed','Churned'], autopct='%1.1f%%',
              colors=['#2ecc71','#e74c3c'], startangle=90, explode=(0,0.06),
              textprops={'fontsize':12})
axes[0,0].set_title('Overall Churn Distribution', fontweight='bold')

# 2. Churn by Geography
geo = df.groupby('Geography')['Exited'].mean().mul(100).reset_index()
geo.columns = ['Country','ChurnRate']
bars = axes[0,1].bar(geo['Country'], geo['ChurnRate'],
                     color=['#3498db','#e67e22','#9b59b6'], edgecolor='white', linewidth=1.2)
axes[0,1].set_title('Churn Rate by Country', fontweight='bold')
axes[0,1].set_ylabel('Churn Rate (%)')
for bar, val in zip(bars, geo['ChurnRate']):
    axes[0,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                   f'{val:.1f}%', ha='center', fontweight='bold')

# 3. Age distribution
axes[0,2].hist(df[df['Exited']==0]['Age'], bins=30, alpha=0.7,
               color='#2ecc71', label='Stayed')
axes[0,2].hist(df[df['Exited']==1]['Age'], bins=30, alpha=0.7,
               color='#e74c3c', label='Churned')
axes[0,2].set_title('Age Distribution vs Churn', fontweight='bold')
axes[0,2].set_xlabel('Age')
axes[0,2].set_ylabel('Count')
axes[0,2].legend()

# 4. Balance boxplot
stayed_bal  = df[df['Exited']==0]['Balance']
churned_bal = df[df['Exited']==1]['Balance']
bp = axes[1,0].boxplot([stayed_bal, churned_bal], tick_labels=['Stayed','Churned'],
                        patch_artist=True,
                        boxprops=dict(facecolor='#3498db', alpha=0.6),
                        medianprops=dict(color='red', linewidth=2))
axes[1,0].set_title('Account Balance vs Churn', fontweight='bold')
axes[1,0].set_ylabel('Balance ($)')

# 5. Active member vs churn
active = df.groupby('IsActiveMember')['Exited'].mean().mul(100).reset_index()
active.columns = ['Active','ChurnRate']
active['Label'] = active['Active'].map({0:'Inactive',1:'Active'})
axes[1,1].bar(active['Label'], active['ChurnRate'],
              color=['#e74c3c','#2ecc71'], edgecolor='white')
axes[1,1].set_title('Churn Rate: Active vs Inactive Members', fontweight='bold')
axes[1,1].set_ylabel('Churn Rate (%)')
for i, row in active.iterrows():
    axes[1,1].text(i, row['ChurnRate']+0.3, f"{row['ChurnRate']:.1f}%",
                   ha='center', fontweight='bold')

# 6. Credit Score distribution
axes[1,2].hist(df[df['Exited']==0]['CreditScore'], bins=30, alpha=0.7,
               color='#2ecc71', label='Stayed')
axes[1,2].hist(df[df['Exited']==1]['CreditScore'], bins=30, alpha=0.7,
               color='#e74c3c', label='Churned')
axes[1,2].set_title('Credit Score Distribution vs Churn', fontweight='bold')
axes[1,2].set_xlabel('Credit Score')
axes[1,2].set_ylabel('Count')
axes[1,2].legend()

plt.tight_layout()
plt.savefig("outputs/01_eda_charts.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ EDA charts saved → outputs/01_eda_charts.png")


# ── STEP 5: Data Preprocessing ───────────────────────────────
print("\n" + "─" * 58)
print("🔧 STEP 4: Preprocessing Data...")

# Drop useless columns
df_model = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# Encode Gender and Geography
le = LabelEncoder()
df_model['Gender']    = le.fit_transform(df_model['Gender'])      # Female=0, Male=1
df_model['Geography'] = le.fit_transform(df_model['Geography'])   # France=0, Germany=1, Spain=2

# Features and target
X = df_model.drop(columns=['Exited'])
y = df_model['Exited']

# Scale numeric features (important for Logistic Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Train / Test split — 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   ✅ Encoded: Gender, Geography")
print(f"   ✅ Scaled : All numeric features")
print(f"   ✅ Train  : {X_train.shape[0]:,} samples")
print(f"   ✅ Test   : {X_test.shape[0]:,} samples")
print(f"   ✅ Features: {X.shape[1]}")


# ── STEP 6: Train Models ──────────────────────────────────────
print("\n" + "─" * 58)
print("🤖 STEP 5: Training ML Models...")

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc  = accuracy_score(y_test, lr_pred)
lr_auc  = roc_auc_score(y_test, lr.predict_proba(X_test)[:,1])

# Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc  = accuracy_score(y_test, rf_pred)
rf_auc  = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])

print(f"\n   Model                  Accuracy    AUC-ROC")
print(f"   ─────────────────────────────────────────")
print(f"   Logistic Regression    {lr_acc*100:.2f}%     {lr_auc:.4f}")
print(f"   Random Forest ✅       {rf_acc*100:.2f}%     {rf_auc:.4f}")


# ── STEP 7: Detailed Evaluation ───────────────────────────────
print("\n" + "─" * 58)
print("📋 STEP 6: Detailed Evaluation (Random Forest)...")
print()
print(classification_report(y_test, rf_pred,
      target_names=['Stayed (0)', 'Churned (1)'], digits=3))


# ── STEP 8: Result Visualizations ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Random Forest — Model Evaluation Results",
             fontsize=15, fontweight='bold')

# 1. Confusion Matrix
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Stayed','Churned'],
            yticklabels=['Stayed','Churned'],
            annot_kws={'size':14})
axes[0].set_title('Confusion Matrix', fontweight='bold')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# 2. ROC Curve
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr.predict_proba(X_test)[:,1])
axes[1].plot(fpr_rf, tpr_rf, color='#e74c3c', lw=2.5,
             label=f'Random Forest  (AUC = {rf_auc:.3f})')
axes[1].plot(fpr_lr, tpr_lr, color='#3498db', lw=2.5,
             label=f'Logistic Reg   (AUC = {lr_auc:.3f})')
axes[1].plot([0,1],[0,1],'k--', lw=1.2, label='Random Guess')
axes[1].fill_between(fpr_rf, tpr_rf, alpha=0.07, color='#e74c3c')
axes[1].set_title('ROC Curve — Model Comparison', fontweight='bold')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(fontsize=10)

# 3. Feature Importance
importances = pd.Series(rf.feature_importances_, index=X.columns)
top = importances.nlargest(10).sort_values()
colors_feat = ['#e74c3c' if i >= top.quantile(0.7) else '#3498db' for i in top]
top.plot(kind='barh', ax=axes[2], color=colors_feat)
axes[2].set_title('Top 10 Feature Importances', fontweight='bold')
axes[2].set_xlabel('Importance Score')
axes[2].axvline(top.mean(), color='gray', linestyle='--', alpha=0.7, label='Mean')
axes[2].legend()

plt.tight_layout()
plt.savefig("outputs/02_model_results.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Model charts saved → outputs/02_model_results.png")


# ── STEP 9: Predict New Customer ─────────────────────────────
print("\n" + "─" * 58)
print("🔮 STEP 7: Predicting for New Customers...")

# Helper function to predict any customer
def predict_customer(credit_score, geography, gender, age, tenure,
                     balance, num_products, has_cr_card,
                     is_active_member, estimated_salary):
    geo_map    = {'France': 0, 'Germany': 1, 'Spain': 2}
    gender_map = {'Female': 0, 'Male': 1}
    raw = pd.DataFrame([{
        'CreditScore'    : credit_score,
        'Geography'      : geo_map[geography],
        'Gender'         : gender_map[gender],
        'Age'            : age,
        'Tenure'         : tenure,
        'Balance'        : balance,
        'NumOfProducts'  : num_products,
        'HasCrCard'      : has_cr_card,
        'IsActiveMember' : is_active_member,
        'EstimatedSalary': estimated_salary
    }])
    scaled = scaler.transform(raw)
    pred   = rf.predict(scaled)[0]
    proba  = rf.predict_proba(scaled)[0]
    return pred, proba

# --- Customer A: High risk profile ---
pred, prob = predict_customer(
    credit_score=400, geography='Germany', gender='Female',
    age=52, tenure=1, balance=140000.0, num_products=1,
    has_cr_card=1, is_active_member=0, estimated_salary=60000
)
print(f"\n   👤 Customer A — High Risk Profile")
print(f"      Credit: 400 | Age: 52 | Germany | Inactive | Balance: $140,000")
print(f"      🎯 Prediction : {'⚠️  WILL CHURN' if pred==1 else '✅  WILL STAY'}")
print(f"      📊 Probability: Stay={prob[0]*100:.1f}% | Churn={prob[1]*100:.1f}%")

# --- Customer B: Low risk profile ---
pred2, prob2 = predict_customer(
    credit_score=750, geography='France', gender='Male',
    age=35, tenure=6, balance=50000.0, num_products=2,
    has_cr_card=1, is_active_member=1, estimated_salary=95000
)
print(f"\n   👤 Customer B — Low Risk Profile")
print(f"      Credit: 750 | Age: 35 | France | Active | 2 Products")
print(f"      🎯 Prediction : {'⚠️  WILL CHURN' if pred2==1 else '✅  WILL STAY'}")
print(f"      📊 Probability: Stay={prob2[0]*100:.1f}% | Churn={prob2[1]*100:.1f}%")


# ── FINAL SUMMARY ─────────────────────────────────────────────
print("\n" + "=" * 58)
print("   ✅ PROJECT COMPLETE — SUMMARY")
print("=" * 58)
print(f"   Dataset          : Churn_Modelling.csv")
print(f"   Total Customers  : {len(df):,}")
print(f"   Features Used    : {X.shape[1]}")
print(f"   Best Model       : Random Forest (200 trees)")
print(f"   Accuracy         : {rf_acc*100:.2f}%")
print(f"   ROC-AUC Score    : {rf_auc:.4f}")
print("=" * 58)
print("\n   📁 Output Files:")
print("      → outputs/01_eda_charts.png")
print("      → outputs/02_model_results.png")

print("=" * 58)
