# 🚀 Credit Card Fraud Detection: A Machine Learning Approach


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

## 🧠 Machine Learning Pipeline

### Data Preprocessing
- **Scaled `Amount`** using `RobustScaler` and **`Time`** using `StandardScaler`.
- Performed **PCA** for dimensionality reduction.
- Applied clustering using **KMeans**.

### Models and Results
| Model                | ROC AUC | Precision | Recall | F1 Score |
|----------------------|---------|-----------|--------|----------|
| Logistic Regression  | 0.96    | 0.87      | 0.85   | 0.86     |
| Random Forest        | 0.97    | 0.90      | 0.88   | 0.89     |
| Gradient Boosting    | 0.96    | 0.88      | 0.86   | 0.87     |
| SVM                  | 0.94    | 0.85      | 0.82   | 0.83     |
| XGBoost              | 0.98    | 0.92      | 0.90   | 0.91     |

### Data Augmentation Results
- Techniques: **SMOTE**, **Polynomial Fit SMOTE Mesh**.
- Improved **Recall** and **Precision** by balancing the dataset.

## 🧑‍💻 Neural Network Implementation

### Architecture
- **Input Layer**
- **Dense Layers** (16 neurons, ReLU activation)
- **Batch Normalization**
- **Output Layer** (Sigmoid activation)

### Training
```python
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64)







