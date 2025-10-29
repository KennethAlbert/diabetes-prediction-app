# 🩺 Diabetes Prediction App

## 📋 Project Overview

A machine learning web application that predicts the likelihood of diabetes based on patient health metrics using a Support Vector Machine (SVM) classifier. This project demonstrates the end-to-end process of building, training, and deploying a medical AI application.

**🎯 SDG Alignment:** This project aligns with **SDG 3: Good Health and Well-being** by providing an accessible tool for early diabetes risk assessment.

## 🚀 Live Demo

[![Streamlit App](https://diabetes-prediction-app-gkd9j8d2hmpnoev4cuuvtn.streamlit.app)

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Training Accuracy** | 78.66% |
| **Testing Accuracy** | 77.27% |
| **Algorithm** | Support Vector Machine (Linear Kernel) |
| **Dataset** | Pima Indians Diabetes Dataset |

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Model Persistence:** Joblib
- **Deployment:** Streamlit Cloud

## 📁 Project Structure

```
diabetes-prediction-app/
│
├── app.py                 # Main Streamlit application
├── train.py              # Model training script
├── requirements.txt       # Python dependencies
├── models/               # Saved ML models
│   ├── scaler.pkl        # StandardScaler object
│   └── classifier.pkl    # Trained SVM model
├── diabetes.csv      # Dataset (not included in repo)
│ 
└── README.md             # Project documentation
```

## 🏃‍♂️ Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction-app.git
   cd diabetes-prediction-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up project structure**
   ```bash
   mkdir models
   mkdir data
   # Place diabetes.csv in the data/ folder
   ```

4. **Train the model**
   ```bash
   python train.py
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📊 Features Used for Prediction

The model uses 8 key health metrics to predict diabetes risk:

| Feature | Description |
|---------|-------------|
| **Pregnancies** | Number of times pregnant |
| **Glucose** | Plasma glucose concentration |
| **Blood Pressure** | Diastolic blood pressure (mm Hg) |
| **Skin Thickness** | Triceps skin fold thickness (mm) |
| **Insulin** | 2-Hour serum insulin (mu U/ml) |
| **BMI** | Body mass index |
| **Diabetes Pedigree Function** | Diabetes likelihood based on family history |
| **Age** | Age in years |

## 🎯 How to Use

1. **Input Patient Data**: Enter the 8 health metrics in the sidebar
2. **Get Prediction**: Click the "Check Diabetes Risk" button
3. **View Results**: 
   - ✅ **Green**: Low risk of diabetes
   - ❌ **Red**: High risk of diabetes (consult healthcare professional)

## 🔬 Model Development

### Data Preprocessing
- Standardization using StandardScaler
- Train-test split (80-20)
- Stratified sampling to maintain class distribution

### Machine Learning
- **Algorithm**: Support Vector Machine with linear kernel
- **Validation**: Cross-validation and accuracy metrics
- **Performance**: 77.27% accuracy on test data

## 🌐 Deployment

This app is deployed on Streamlit Cloud with the following configuration:

1. **Repository**: Connected to GitHub
2. **Main file**: `app.py`
3. **Python version**: 3.8+
4. **Dependencies**: Managed via `requirements.txt`

## ⚠️ Important Disclaimer

> **This application is for educational and demonstration purposes only.** 
> 
> - ❌ Not a medical diagnostic tool
> - ❌ Should not replace professional medical advice
> - ❌ Results should be verified by healthcare professionals
> - ✅ Always consult with qualified medical practitioners for health concerns

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Model improvements
- UI/UX enhancements
- Additional features
- Bug fixes

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset: Pima Indians Diabetes Dataset
- Scikit-learn library for machine learning tools
- Streamlit for the web framework
- Icons: Twemoji

## 📞 Support

For questions or support:

- 📧 Email: your-email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/diabetes-prediction-app/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/diabetes-prediction-app/discussions)

---

**Made with ❤️ for better healthcare accessibility**
