# Home Credit Default Risk – Comparative Machine Learning & Deep Learning Study

This repository presents a comprehensive end-to-end workflow for credit default risk prediction using the [Home Credit Default Risk dataset](https://www.kaggle.com/competitions/home-credit-default-risk). The project systematically compares classical machine learning (ML) and deep learning (DL) models under a unified pipeline, covering data preprocessing, feature engineering, class imbalance handling, model training, and evaluation.

---

## 📌 Project Motivation

Accurately predicting credit default risk is crucial for financial institutions to minimize losses and make informed lending decisions. Traditional methods often fall short in capturing the complex relationships present in high-dimensional financial data. This project aims to benchmark a range of ML and DL models on a real-world dataset, exploring the impact of data preprocessing and model selection on predictive performance.

---

## 🎯 Objectives

- **To build an end-to-end pipeline** for credit default risk prediction.
- **To handle class imbalance** via both algorithmic and sampling strategies.
- **To train and compare multiple ML & DL models** using standardized evaluation metrics.
- **To analyze the strengths and weaknesses** of each approach and provide recommendations for real-world applications.

---

## 🗃️ Dataset

- **Source:** [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/competitions/home-credit-default-risk)
- **Samples:** 307,511 applications
- **Features:** 122 numerical and categorical variables
- **Target:** Binary (0 = no default, 1 = default)

---

## ⚙️ Workflow & Methodology

1. **Data Preprocessing**
   - Removal of features with excessive missing values
   - Imputation of missing data (median for numerical, mode for categorical)
   - Encoding categorical features (Label Encoding, One-Hot Encoding)
   - Feature scaling with StandardScaler

2. **Feature Selection**
   - Univariate selection (Chi-square test)
   - Recursive Feature Elimination (RFE)
   - Model-based importance (tree-based methods)

3. **Handling Class Imbalance**
   - Use of class weights in algorithms
   - Oversampling (SMOTE) where applicable

4. **Model Training**
   - **Machine Learning Models:** Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost
   - **Deep Learning Models:** DNN (MLP), CNN, LSTM

5. **Model Evaluation**
   - 5-fold stratified cross-validation for robust comparison
   - Metrics: Accuracy, Precision, Recall, F1-score, ROC AUC
   - Visualization: Confusion matrix, ROC curve, feature importance

---

## 🏆 Key Results

- **Best ML Model:** XGBoost achieved the highest AUC and recall, making it ideal for detecting potential defaults.
- **Best DL Model:** DNN (MLP) provided competitive precision but overall lower recall due to class imbalance.
- **Overall:** Classical gradient boosting models (XGBoost, CatBoost) outperformed others in most scenarios. Deep learning models showed promise, but required further optimization for tabular, imbalanced datasets.

---

## 📄 Files in this Repository

- `Home_Credit_Default_Risk_Report.ipynb` – Main Colab/Jupyter notebook (Turkish; code, visualizations, and analysis)
- `Home_Credit_Default_Risk_IEEE_Paper.pdf` – Final IEEE-format project paper (English)
- `README.md` – Project overview and usage instructions

---

## 🚀 How to Use

1. Download or clone the repository.
2. Open `Home_Credit_Default_Risk_Report.ipynb` in [Google Colab](https://colab.research.google.com/) or Jupyter Notebook.
3. Follow the step-by-step workflow for data loading, preprocessing, model training, and evaluation.
4. For a concise summary and academic presentation, refer to the `Home_Credit_Default_Risk_IEEE_Paper.pdf`.

---

## 📊 Sample Outputs

- Model performance comparison charts
- ROC curves and confusion matrices for each model
- Feature importance visualizations

---

## 💡 Key Insights & Discussion

- Handling class imbalance is **crucial** for fair evaluation—recall and F1-score are as important as accuracy.
- Ensemble ML models (XGBoost, CatBoost) are strong baselines for tabular financial data.
- Deep learning methods need careful tuning and possibly more advanced architectures for such data.
- Feature selection and robust cross-validation are vital for model generalization.

---

## 👤 Author

**Arda Adar**  
Computer Engineering Department, Biruni University, Istanbul, Türkiye  
Contact: 220404013@st.biruni.edu.tr

---

## 📚 References

- [Project Kaggle page](https://www.kaggle.com/competitions/home-credit-default-risk)
- See PDF report for detailed references.

---

*For more details and the full workflow, see the notebook and the paper included in this repository!*
