# 🩺 Hypertension Risk Prediction Analysis

## 📌 Project Overview

This comprehensive machine learning project analyzes the **Hypertension Risk Prediction Dataset** from Kaggle to predict hypertension risk in patients. The analysis encompasses exploratory data analysis, feature engineering, multiple machine learning models, and clinical insights for healthcare applications.

**Dataset Source**: [Kaggle - Hypertension Risk Prediction Dataset](https://www.kaggle.com/datasets/miadul/hypertension-risk-prediction-dataset/data)

## 🎯 Project Objectives

- 🔍 **Exploratory Data Analysis (EDA)**: Understand risk factors and patterns in hypertension development
- 🤖 **Binary Classification Modeling**: Predict hypertension risk using multiple machine learning algorithms
- 📊 **Feature Importance Analysis**: Identify key factors influencing hypertension risk
- 🎯 **Model Evaluation**: Compare performance using accuracy, F1-score, ROC-AUC metrics
- 🏥 **Clinical Application**: Develop a practical prediction function for healthcare use

## 📁 Dataset Description

The dataset contains **1,985 patient records** with **11 features** related to hypertension risk factors:

### Features:
- **Age**: Patient's age in years
- **Salt_Intake**: Daily salt intake in grams
- **Stress_Score**: Psychological stress level (0-10 scale)
- **BP_History**: Previous blood pressure status (Normal, Prehypertension, Hypertension)
- **Sleep_Duration**: Average sleep hours per day
- **BMI**: Body Mass Index
- **Medication**: Current medication type (None, ACE Inhibitor, Beta Blocker, Other)
- **Family_History**: Family history of hypertension (Yes/No)
- **Exercise_Level**: Physical activity level (Low, Moderate, High)
- **Smoking_Status**: Smoking habits (Non-Smoker, Smoker)
- **Has_Hypertension**: Target variable (Yes/No)

### Dataset Characteristics:
- **Balanced Dataset**: 52% hypertension cases vs 48% no hypertension
- **No Missing Values**: Complete dataset ready for analysis
- **Real-world Relevance**: Features align with clinical risk factors

## 🚀 Key Findings

### Model Performance
- **Best Model**: XGBoost achieved **97.7% accuracy** and **99.8% ROC-AUC**
- **Robust Performance**: All models showed excellent performance with proper preprocessing
- **Clinical Validation**: Results align with established medical research

### Primary Risk Factors (Feature Importance)
1. **Blood Pressure History** (38% importance) - Most significant predictor
2. **Smoking Status** (12% importance) - Strong lifestyle factor
3. **Family History** (10% importance) - Genetic predisposition
4. **Age** (10% importance) - Natural aging factor
5. **Stress Score** (10% importance) - Psychological impact

### Clinical Insights
- **High Risk Profile**: Age >60, Salt >10g/day, Stress >7, BMI >30, Previous BP issues, Smoking, Family history
- **Low Risk Profile**: Age <30, Salt <7g/day, Stress <3, BMI <25, Normal BP history, Non-smoker
- **Risk Stratification**: Model provides clear probability scores for clinical decision-making

## 🛠️ Technologies Used

### Data Analysis & Visualization
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib & seaborn**: Static visualizations
- **plotly**: Interactive visualizations

### Machine Learning
- **scikit-learn**: Core ML algorithms and preprocessing
  - Logistic Regression
  - Random Forest Classifier
  - StandardScaler, LabelEncoder
- **XGBoost**: Gradient boosting algorithm
- **SHAP**: Model interpretability (when available)

### Development Environment
- **Jupyter Notebook**: Interactive development and analysis
- **Python 3.x**: Programming language

## 📊 Project Structure

```
Hypertension-Risk-Prediction-Analysis/
│
├── hypertension_dataset.csv                    # Dataset from Kaggle
├── hypertension_risk_prediction_analysis.ipynb # Main analysis notebook
└── README.md                                   # Project documentation
```

## 🔬 Analysis Workflow

1. **Data Loading & Exploration**
   - Dataset overview and structure analysis
   - Missing value and duplicate detection
   - Target variable distribution analysis

2. **Exploratory Data Analysis**
   - Statistical summaries and distributions
   - Correlation analysis between features
   - Visualization of risk factors by hypertension status

3. **Data Preprocessing**
   - Categorical variable encoding
   - Feature scaling and normalization
   - Train-test split preparation

4. **Model Development**
   - Multiple algorithm implementation
   - Hyperparameter tuning with GridSearchCV
   - Cross-validation for robust evaluation

5. **Model Evaluation**
   - Performance metrics comparison
   - ROC curve and confusion matrix analysis
   - Feature importance visualization

6. **Clinical Application**
   - Prediction function development
   - Risk stratification examples
   - Healthcare recommendations

## 🏥 Clinical Applications

### Risk Assessment Protocol
The developed model can be integrated into clinical workflows to:
- **Early Detection**: Identify high-risk patients before hypertension develops
- **Resource Optimization**: Focus interventions on highest-risk individuals
- **Prevention Programs**: Design targeted lifestyle modification programs
- **Clinical Decision Support**: Provide probability-based risk assessments

### Deployment-Ready Function
The notebook includes a `predict_hypertension_risk()` function that:
- Accepts standard clinical parameters
- Returns both risk category and probability score
- Includes all necessary preprocessing steps
- Can be integrated into Electronic Health Records (EHR)

## 📈 Results Summary

### Model Comparison
| Model | Accuracy | F1-Score | ROC-AUC | Key Strengths |
|-------|----------|----------|---------|---------------|
| Logistic Regression | ~95% | ~95% | ~98% | Interpretable, fast |
| Random Forest | ~96% | ~96% | ~99% | Feature importance, robust |
| **XGBoost** | **97.7%** | **97.8%** | **99.8%** | **Best performance, handles complex patterns** |

### Business Impact
- **Preventive Care**: Enable proactive hypertension prevention
- **Cost Reduction**: Reduce long-term cardiovascular treatment costs
- **Quality Improvement**: Enhance patient outcomes through early intervention
- **Efficiency**: Optimize healthcare resource allocation

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost
```

### Running the Analysis
1. Clone this repository
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/miadul/hypertension-risk-prediction-dataset/data)
3. Place `hypertension_dataset.csv` in the project directory
4. Open `hypertension_risk_prediction_analysis.ipynb` in Jupyter Notebook
5. Run all cells to reproduce the analysis

## 🔮 Future Enhancements

- **Extended Dataset**: Incorporate additional clinical parameters
- **Temporal Analysis**: Track risk factor changes over time
- **Model Ensemble**: Combine multiple algorithms for improved accuracy
- **Web Application**: Develop user-friendly interface for clinicians
- **Real-time Integration**: Connect with hospital information systems
- **Validation Studies**: Test model on external clinical datasets

## 📄 License

This project is available under the MIT License. The dataset is provided by Kaggle under its respective terms.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve this analysis.

## 📞 Contact

For questions or collaboration opportunities, please open an issue in this repository.

---

**Note**: This analysis is for educational and research purposes. Any clinical implementation should involve medical professionals and proper validation studies.
