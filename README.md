# 🚀 Credit Card Fraud Detection: A Machine Learning Approach

![Project Banner](https://via.placeholder.com/1200x300.png?text=Credit+Card+Fraud+Detection+%F0%9F%94%92)

## 📚 Introduction
Credit card fraud detection is critical for financial institutions to mitigate losses and protect customer assets. This project leverages machine learning models to detect fraudulent transactions using the Kaggle [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).

## 🔍 Dataset Overview
The dataset contains transactions made by European cardholders in September 2013. It consists of 284,807 transactions, where only 492 are fraudulent (~0.172% of all transactions).

- **Features:** 30 numerical features (V1-V28 from PCA transformation, `Time`, and `Amount`).
- **Target:** `Class` (1 for fraud, 0 for non-fraud).

---

## 🎯 Project Highlights
1. Data preprocessing (scaling, augmentation, and sampling).
2. Exploratory Data Analysis (EDA) with interactive visuals.
3. Machine learning models:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - Support Vector Machines (SVM)
   - XGBoost
4. Neural Network using TensorFlow.
5. Data augmentation techniques (e.g., SMOTE variants).
6. Grid search for hyperparameter tuning.

---

## 📊 Exploratory Data Analysis (EDA)

### Fraud Distribution
```python
sns.countplot(x=data['Class'], palette='coolwarm')


### Feature Correlation
```python
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True)
![Correlation Matrix](https://via.placeholder.com/800x400.png?text=Correlation+Matrix)


