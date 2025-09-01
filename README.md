# Steel Plate Defect Classification for Predictive-Maintenance

[![Kaggle](https://img.shields.io/badge/Kaggle-View%20Notebook-blue?logo=kaggle)](https://www.kaggle.com/code/sulymansifat/steel-plates-faults-classification-pdm)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)

A supervised machine learning classification project for Coursera course analyzing steel plate defects using UCI dataset (1,941 samples). Compared Logistic Regression, Random Forest & SVM algorithms. Random Forest achieved 80.7% accuracy for automated quality control. Demonstrates 64.6% cost savings potential in manufacturing.

## 🎯 Project Overview

This project applies machine learning classification techniques to predict steel plate manufacturing defects, supporting smart manufacturing and predictive maintenance objectives. The analysis demonstrates how automated quality control can reduce manual inspection costs while improving product quality.

### 🏆 Key Results
- **Best Model**: Random Forest with **80.7% accuracy**
- **Precision**: 78.3% (minimal false alarms)
- **Recall**: 61.5% (defect detection rate)
- **Cost Savings**: 64.6% potential reduction in inspection costs
- **Critical Feature**: Steel_Plate_Thickness (8.0% importance)

## 📊 Dataset

- **Source**: [UCI Steel Plates Faults Dataset](https://archive.ics.uci.edu/ml/datasets/Steel+Plates+Faults)
- **Samples**: 1,941 steel plates
- **Features**: 27 geometric and luminosity measurements
- **Target**: 7 types of manufacturing faults
- **Quality**: Zero missing values

### Dataset Files
```
Steel Plates Faults Dataset (UCI)/
├── Faults.NNA          # Main dataset
└── Faults27x7_var      # Feature names
```

## 🔬 Methodology

### Models Implemented
1. **Logistic Regression** (Baseline - High Interpretability)
2. **Random Forest** (Ensemble - Balanced Performance) ⭐
3. **Support Vector Machine** (Advanced - Complex Patterns)

### Analysis Pipeline
1. **Data Exploration**: Feature distributions, correlations, outlier analysis
2. **Preprocessing**: Target engineering, train-test split (80/20), feature scaling
3. **Model Training**: Cross-validation with consistent methodology
4. **Evaluation**: Comprehensive metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
5. **Business Analysis**: Cost-benefit evaluation and implementation recommendations

## 📈 Results Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 68.6% | 62.3% | 24.4% | 35.1% | 67.4% |
| **Random Forest** | **80.7%** | **78.3%** | **61.5%** | **68.9%** | **87.2%** |
| Support Vector Machine | 76.6% | 73.9% | 50.4% | 59.9% | 79.7% |

### 🔍 Key Insights
- **Steel_Plate_Thickness** emerged as the most predictive feature
- **34.7% defect rate** in manufacturing process
- **Geometric measurements** provide strong defect indicators
- **Area relationships** enhance prediction accuracy

## 💼 Business Impact

- **Manual Inspection Cost**: $9,705 for 1,941 samples
- **Automated Accuracy**: 80.7% with Random Forest
- **Potential Savings**: 64.6% cost reduction (~$6,267 annually)
- **Quality Improvement**: 61.5% defect detection rate

## 📁 Repository Structure

```
steel-plate-defect-classification/
├── README.md                                    # Project overview
├── steel-plates-faults-classification-pdm.ipynb # Jupyter notebook
├── Steel Plates Faults Dataset (UCI)/          # Dataset folder
│   ├── Faults.NNA                             # Main dataset
│   └── Faults27x7_var                         # Feature names
└── Steel Plates Faults Classification for Smart Manufacturing_ A Predictive Maintenance Approach.pdf
                                                # Complete project report
```

## 📄 Documentation

### 📊 Interactive Analysis
- **Kaggle Notebook**: [View Live Analysis](https://www.kaggle.com/code/sulymansifat/steel-plates-faults-classification-pdm)
- **Code Implementation**: Complete analysis pipeline with visualizations

### 📋 Comprehensive Report
- **PDF Report**: `Steel Plates Faults Classification for Smart Manufacturing_ A Predictive Maintenance Approach.pdf`
- **Academic Format**: Detailed methodology, results, and business recommendations
- **Stakeholder Ready**: Suitable for Chief Data Officer / Head of Analytics

## 🛠️ Technologies Used

- **Python 3.8+**
- **Machine Learning**: scikit-learn
- **Data Analysis**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Environment**: Kaggle Notebooks

### Dependencies
```python
pandas
numpy
scikit-learn
matplotlib
seaborn
```

## 🚀 Getting Started

### Option 1: Kaggle (Recommended)
1. Visit the [Kaggle notebook](https://www.kaggle.com/code/sulymansifat/steel-plates-faults-classification-pdm)
2. Click "Copy & Edit" to run your own version
3. All dependencies and dataset are pre-loaded

### Option 2: Local Setup
1. Clone this repository
2. Install required packages: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Run the Jupyter notebook: `jupyter notebook steel-plates-faults-classification-pdm.ipynb`

## 📊 Key Visualizations

The analysis includes 9 comprehensive visualizations:
- Fault distribution analysis
- Feature correlation heatmaps
- Model performance comparisons
- Confusion matrices for all models
- Feature importance rankings
- Business impact analysis

## 🔮 Future Enhancements

- **Expanded Dataset**: Additional process parameters (temperature, pressure)
- **Multi-Class Models**: Separate models for each fault type
- **Real-Time Integration**: Manufacturing system deployment
- **Deep Learning**: Neural networks for complex pattern recognition
- **Temporal Analysis**: Equipment wear progression modeling

## 📧 Contact

**Author**: Md. Sulyman Islam Sifat  
**Course**: Coursera - Supervised Machine Learning: Classification  
**Date**: September 2, 2025

---


### 🙏 Acknowledgments

- UCI Machine Learning Repository for the Steel Plates Faults Dataset
- Coursera for the excellent Supervised Machine Learning course
- Manufacturing industry domain experts for business context validation

---

⭐ **Star this repository if you found it helpful!**


