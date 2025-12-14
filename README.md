# Customer Churn Prediction with Machine Learning

A comprehensive machine learning project predicting bank customer churn with 86.80% accuracy, implementing multiple algorithms, explainable AI (SHAP), and interactive prediction capabilities.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Key Results](#key-results)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Key Insights](#key-insights)
- [Visualizations](#visualizations)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

This project addresses the critical business challenge of customer churn in the banking sector. Using machine learning techniques, the system identifies customers at risk of leaving, enabling proactive retention strategies.

### Problem Statement
Customer churn costs banks billions annually. This project builds a predictive system to:
- Identify high-risk customers before they leave
- Understand key drivers of customer churn
- Provide actionable insights for retention campaigns
- Enable data-driven decision making

### Solution
An end-to-end machine learning pipeline that:
- Analyzes customer behavior patterns
- Trains and compares multiple ML algorithms
- Provides explainable predictions using SHAP
- Delivers 86.80% accuracy in churn prediction

---

## 📊 Dataset

**Source:** [Bank Customer Churn Dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset) (Kaggle)

**Specifications:**
- **Total Customers:** 10,000
- **Features:** 12 (11 used for modeling)
- **Target Variable:** Binary classification (0 = Retained, 1 = Churned)
- **Class Distribution:** 79.6% Retained, 20.4% Churned
- **Data Quality:** No missing values (100% complete)

**Features Used:**
- `credit_score`: Customer credit score (300-850)
- `gender`: Customer gender
- `age`: Customer age
- `tenure`: Years with the bank
- `balance`: Account balance
- `products_number`: Number of bank products used
- `credit_card`: Credit card ownership
- `active_member`: Active membership status
- `estimated_salary`: Annual salary estimate
- `country`: Customer location (France, Germany, Spain)
- `churn`: Target variable (0/1)

---

## 🏆 Key Results

### Model Comparison

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score |
|-------|----------|-------------------|----------------|----------|
| Logistic Regression | 80.80% | 58.91% | 18.67% | 0.28 |
| **Random Forest** | **86.80%** | **81.50%** | **45.45%** | **0.58** |
| XGBoost | 86.45% | 75.00% | 50.12% | 0.60 |

### Best Model: Random Forest
- **Accuracy:** 86.80%
- **Correctly Identified Churned Customers:** 185 out of 407 (45.45% recall)
- **Precision:** 81.50% (minimizes false alarms)
- **Business Impact:** Can proactively target ~185 at-risk customers with 81% confidence

---

## 🛠️ Technologies Used

### Programming Language
- Python 3.12

### Core Libraries
- **Data Processing:** Pandas 2.2.2, NumPy 2.0.2
- **Machine Learning:** Scikit-learn 1.6.1, XGBoost 3.1.2
- **Explainability:** SHAP 0.50.0
- **Visualization:** Matplotlib 3.10.0, Seaborn 0.13.2
- **Development:** Jupyter Notebook

### Machine Learning Algorithms
- Logistic Regression (Baseline)
- Random Forest Classifier (Best Performance)
- XGBoost Classifier (High Recall)

### Explainability
- SHAP (SHapley Additive exPlanations)
- Feature Importance Analysis

---

## 📁 Project Structure
```
customer-churn-prediction/
│
├── notebook.ipynb                      # Main Jupyter notebook with complete analysis
│
├── outputs/                            # Generated visualizations and reports
│   ├── churn_distribution.png         # Target variable distribution
│   ├── age_comparison.png             # Age analysis (churned vs retained)
│   ├── feature_importance.png         # Feature importance ranking
│   ├── shap_summary_plot.png          # SHAP explainability plot
│   └── model_performance_report.txt   # Detailed performance metrics
│
├── requirements.txt                    # Python dependencies
├── README.md                          # Project documentation
└── .gitignore                         # Git ignore rules
```

**Note:** Due to GitHub file size limitations:
- **Dataset:** Download from [Kaggle](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)
- **Trained Models:** Run the notebook to regenerate `.pkl` model files
- All code, visualizations, and results are included

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook or Google Colab

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the dataset**
- Visit [Kaggle Dataset Page](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)
- Download `Bank Customer Churn Prediction.csv`
- Place in `data/raw/` directory

4. **Launch Jupyter Notebook**
```bash
jupyter notebook notebook.ipynb
```

Or upload to **Google Colab** for cloud execution.

---

## 💻 Usage

### Running the Complete Analysis

1. Open `notebook.ipynb` in Jupyter or Colab
2. Execute cells sequentially (each step is clearly documented)
3. Models will be trained and saved automatically
4. Visualizations will be generated in `outputs/` folder

### Making Predictions on New Data

The notebook includes an interactive prediction function:
```python
# Example usage
customer_data = {
    'credit_score': 650,
    'gender': 1,  # 0=Female, 1=Male
    'age': 42,
    'tenure': 5,
    'balance': 85000.0,
    'products_number': 2,
    'credit_card': 1,
    'active_member': 1,
    'estimated_salary': 95000.0,
    'country_Germany': 0,
    'country_Spain': 0
}

result = predict_customer_churn(customer_data)
print(f"Churn Probability: {result['churn_probability']:.2%}")
print(f"Risk Level: {result['risk_level']}")
```

---

## 📈 Model Performance

### Training Details
- **Training Set:** 8,000 customers (80%)
- **Testing Set:** 2,000 customers (20%)
- **Cross-Validation:** Stratified split to maintain class balance
- **Preprocessing:** StandardScaler for feature normalization

### Confusion Matrix (Random Forest - Best Model)

|                | Predicted: Retain | Predicted: Churn |
|----------------|-------------------|------------------|
| **Actual: Retain** | 1,551 (TN)        | 42 (FP)          |
| **Actual: Churn**  | 222 (FN)          | 185 (TP)         |

### Metrics Explanation
- **True Negatives (1,551):** Correctly predicted retained customers
- **True Positives (185):** Correctly predicted churned customers
- **False Positives (42):** Incorrectly flagged as churn (low - good!)
- **False Negatives (222):** Missed churned customers (opportunity for improvement)

---

## 💡 Key Insights

### Feature Importance (Random Forest)

| Rank | Feature | Importance | Insight |
|------|---------|------------|---------|
| 1 | Age | 35.76% | Primary churn driver |
| 2 | Products Number | 27.17% | Strong retention indicator |
| 3 | Balance | 8.99% | Moderate impact |
| 4 | Active Member | 7.93% | Critical for retention |
| 5 | Country (Germany) | 6.13% | Geographic patterns |

### Business Insights

1. **Age is the Strongest Predictor**
   - Churned customers are **7.4 years older** on average (44.8 vs 37.4 years)
   - Older customers (45+) show significantly higher churn risk
   - **Action:** Develop age-specific retention programs

2. **Product Engagement Matters**
   - Customers with only 1 product have higher churn rates
   - Multiple product ownership indicates stronger bank relationship
   - **Action:** Cross-selling campaigns to increase product adoption

3. **Activity Status is Critical**
   - Inactive members show dramatically higher churn probability
   - Early warning sign for at-risk customers
   - **Action:** Automated engagement monitoring and alerts

4. **Counterintuitive Finding: High Balance ≠ Loyalty**
   - Churned customers have **higher median balance** ($109K vs $92K)
   - Wealthy customers have more banking options
   - **Action:** Premium service and personalized attention for high-balance accounts

5. **Geographic Patterns**
   - Germany shows higher churn rates than France or Spain
   - Regional factors influence customer behavior
   - **Action:** Region-specific retention strategies

---

## 📊 Visualizations

The project includes comprehensive visualizations:

### 1. Churn Distribution
![Churn Distribution](outputs/churn_distribution.png)
- 20.4% churn rate (realistic for banking industry)
- Class imbalance addressed through stratified sampling

### 2. Age Comparison
![Age Comparison](outputs/age_comparison.png)
- Side-by-side comparison of age distributions
- Clear evidence of age-related churn patterns

### 3. Feature Importance
![Feature Importance](outputs/feature_importance.png)
- Ranked list of predictive features
- Guides business focus areas

### 4. SHAP Summary Plot
![SHAP Analysis](outputs/shap_summary_plot.png)
- Explainable AI showing how features impact predictions
- Transparency for stakeholders and regulatory compliance

---

## 🔮 Future Enhancements

### Technical Improvements
- [ ] Hyperparameter tuning using GridSearchCV
- [ ] Ensemble methods combining multiple models
- [ ] Deep learning approaches (Neural Networks)
- [ ] Real-time prediction API using Flask/FastAPI
- [ ] Model retraining pipeline with MLOps tools

### Business Features
- [ ] Customer Lifetime Value (CLV) prediction
- [ ] Personalized retention strategy generator using LLMs
- [ ] A/B testing framework for retention campaigns
- [ ] Interactive dashboard with Streamlit or Dash
- [ ] Integration with CRM systems (Salesforce, HubSpot)

### Data Enhancements
- [ ] Time-series analysis of churn patterns
- [ ] Additional features (transaction history, customer service interactions)
- [ ] External data integration (economic indicators, competitor analysis)
- [ ] Sentiment analysis from customer feedback

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve this project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

### Areas for Contribution
- Model improvements and new algorithms
- Additional visualizations
- Code optimization
- Documentation enhancements
- Bug fixes

---

## 📄 License

This project is licensed under the MIT License - see below for details:
```
MIT License

Copyright (c) 2025 Snehan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 👤 Contact

**Snehan**

- 🔗 LinkedIn: [LinkedIn Profile](www.linkedin.com/in/snehan-snehan)
- 📧 Email: snehandhanraj@gmail.com


---

## 🙏 Acknowledgments

- **Dataset:** [Kaggle - Bank Customer Churn Dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)
- **Inspiration:** Real-world customer retention challenges in the banking industry
- **Libraries:** Thanks to the open-source community for excellent ML tools
- **Learning Resources:** Scikit-learn documentation, SHAP tutorials, Kaggle community

---

## 📌 Project Status

**Status:** ✅ Complete and Production-Ready

**Last Updated:** December 2025

**Version:** 1.0.0

---

## ⭐ If you found this project helpful, please consider giving it a star!

---

**Keywords:** machine learning, customer churn, prediction, classification, random forest, xgboost, SHAP, explainable AI, banking, data science, python, scikit-learn
