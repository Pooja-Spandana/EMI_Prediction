# 🏦 EMI Prediction Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://emiprediction.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow)](https://mlflow.org/)
[![DagsHub](https://img.shields.io/badge/DagsHub-Integration-blue)](https://dagshub.com/)


### Streamlit Application 

### 🔮 Real-Time Predictions
- **Dual ML Models**: LightGBM for classification (97.16% accuracy) and XGBoost for regression (R² = 0.9817)
- **Instant Results**: Get eligibility status and maximum EMI amount in seconds
- **Confidence Scores**: View prediction confidence levels for transparency

### 📊 Interactive Data Explorer
- **Univariate Analysis**: Explore distributions of numerical and categorical features
- **Bivariate Analysis**: Analyze relationships between features with scatter plots
- **Correlation Heatmaps**: Visualize feature correlations
- **Live Data**: Loads dataset directly from DagsHub via DVC

### 📈 Model Performance Dashboard
- **Comprehensive Metrics**: View accuracy, precision, recall, F1-score, R², RMSE, MAE, and MAPE
- **MLflow Integration**: Track experiments, model versions, and artifacts
- **Model Registry**: Access production models with versioning

### 🎨 Modern UI/UX
- **Dark Theme**: Professional dark mode with vibrant blue accents
- **Responsive Design**: Optimized for desktop and mobile viewing
- **Center-Aligned Headers**: Clean, consistent layout across all pages

---

## � Approach: From Raw Data to Model-Ready Datasets

This section outlines the complete data pipeline and methodology used to transform raw financial data into production-ready machine learning models.

### 1️⃣ Data Ingestion (`data_ingestion.py`)

**Objective**: Load and perform initial validation of raw data

- **Input**: `Data/Raw/emi_prediction_dataset.csv` (~400,000 records)
- **Process**:
  - Load raw CSV data
  - Perform basic data type validation
- **Output**: 
  - `Data/Interim/ingested_raw.csv`

### 2️⃣ Data Cleaning (`data_cleaning.py`)

**Objective**: Handle missing values, outliers, and data quality issues

- **Input**: `Data/Interim/ingested_raw.csv`
- **Process**:
  - **Missing Value Imputation**:
    - Dropped missing values
  - **Outlier Detection & Treatment**:
    - IQR method for numerical features
    - Cap outliers at 99th percentile domain caps
  - **Data Type Corrections**:
    - Extract numeric values from string columns (e.g., "₹ 50,000" → 50000)
    - Convert categorical variables to proper dtype
  - **Duplicate Removal**: Remove exact duplicate records
- **Output**: 
  - `Data/Interim/cleaned_data.csv`

### 3️⃣ Feature Engineering (`feature_engineering.py`)

**Objective**: Create meaningful features and prepare data for modeling

- **Input**: Data/Interim/cleaned_data.csv`
- **Process**:
  - **Derived Features**:
    - Feature engineered 7 new features
    - Dropped redundant features
  - **Split Train/Val/Test**
    - Train - 70%
    - Val - 15%
    - Test - 15%
  - **Categorical Encoding**:
    - One-Hot Encoding for nominal features (Gender, Education, Employment Type, etc.)
    - Label Encoding for target feature
  - **Feature Scaling**:
    - RobustScaler for numerical features
- **Output**: 
  - `Data/Interim/FE_dataset.csv`
  - `Data/Processed/train`
  - `Data/Processed/val`
  - `Data/Processed/test`
  - Final feature count: **42 features** (after preprocessing)

### 4️⃣ Model Training & Evaluation

#### A. Regression Model Training (`reg_trainer.py`)

**Objective**: Predict maximum monthly EMI amount

- **Algorithm**: XGBoost Regressor
- **Hyperparameter Tuning**: RandomizedSearchCV with 3-fold cross-validation
- **Key Parameters**:
  - `n_estimators`: [100, 200, 300]
  - `max_depth`: [3, 5, 7, 10]
  - `learning_rate`: [0.01, 0.05, 0.1]
  - `subsample`: [0.8, 0.9, 1.0]
- **Evaluation Metrics**:
  - R² Score: 0.9817
  - RMSE: 996.55
  - MAE: 612.89
  - MAPE: 8.23%

#### B. Classification Model Training (`cls_trainer.py`)

**Objective**: Predict EMI eligibility status (3 classes)

- **Algorithm**: LightGBM Classifier
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Hyperparameter Tuning**: RandomizedSearchCV with stratified 3-fold CV
- **Key Parameters**:
  - `n_estimators`: [100, 200, 300]
  - `max_depth`: [3, 5, 7]
  - `learning_rate`: [0.01, 0.05, 0.1]
  - `num_leaves`: [31, 50, 70]
- **Evaluation Metrics**:
  - Accuracy: 97.16%
  - Precision: 0.9712
  - Recall: 0.9703
  - F1-Score: 0.9707
  - ROC-AUC: 0.9945

#### C. Final Model Training (`final_trainer.py`)

**Objective**: Train final models on combined train+validation data

- **Process**:
  - Combine train and validation sets for final training
  - Use best hyperparameters from tuning phase
  - Evaluate on held-out test set
  - Log all metrics, parameters, and artifacts to MLflow
  - Register models in MLflow Model Registry
- **Artifacts Saved**:
  - Trained models: `artifacts/models/`
  - Best parameters: `artifacts/best_params/`
  - Preprocessor: `artifacts/preprocessor/`
  - Confusion matrices, feature importance plots

### 5️⃣ Model Deployment Pipeline

- **Preprocessing Pipeline**:
  - Saved as `preprocessor.pkl` using joblib
  - Includes all transformations (scaling, encoding)
  - Ensures consistency between training and inference

- **Model Serving**:
  - Models loaded from MLflow Model Registry
  - Fallback to local artifacts if MLflow unavailable
  - Real-time predictions via Streamlit interface

- **Prediction Workflow**:
  1. User inputs raw features via web form
  2. Apply feature engineering transformations
  3. Preprocess using saved preprocessor
  4. Generate predictions using both models
  5. Return eligibility status + maximum EMI amount

---

## 📊 Model Performance

### Classification Model (LightGBM)
| Metric        | Score  |
| ------------- | ------ |
| **Accuracy**  | 97.16% |
| **Precision** | 0.9712 |
| **Recall**    | 0.9703 |
| **F1-Score**  | 0.9707 |
| **ROC-AUC**   | 0.9945 |

### Regression Model (XGBoost)
| Metric       | Score  |
| ------------ | ------ |
| **R² Score** | 0.9817 |
| **RMSE**     | 996.55 |
| **MAE**      | 612.89 |
| **MAPE**     | 8.23%  |

---

### 🔗 Quick Links
- **Live App**: [https://emiprediction.streamlit.app](https://emiprediction.streamlit.app)
- **DagsHub Repository**: [https://dagshub.com/Pooja-Spandana/EMI_Prediction](https://dagshub.com/Pooja-Spandana/EMI_Prediction)
- **MLflow Experiments**: [View Experiment Tracking](https://dagshub.com/Pooja-Spandana/EMI_Prediction.mlflow/#/compare-experiments/s?experiments=%5B%220%22%2C%222%22%5D&searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D)

---

## �🛠️ Tech Stack

### Machine Learning
- **XGBoost** - Regression model for EMI amount prediction
- **LightGBM** - Classification model for eligibility assessment
- **Scikit-learn** - Data preprocessing and feature engineering
- **SMOTE** - Handling class imbalance in training data

### MLOps & Tracking
- **MLflow** - Experiment tracking and model registry
- **DagsHub** - Remote MLflow tracking server and collaboration
- **DVC** - Data version control and management

### Web Application
- **Streamlit** - Interactive web application framework
- **Plotly** - Interactive data visualizations
- **Pandas & NumPy** - Data manipulation and analysis

### Deployment
- **Streamlit Cloud** - Cloud hosting and deployment
- **GitHub** - Version control and CI/CD

---

## 📁 Project Structure

```
EMI_Prediction/
├── .dvc/                                  # DVC configuration
├── .streamlit/                            # Streamlit configuration
│   └── config.toml                        # Theme and UI settings
├── Data/
│   ├── Raw/                               # Raw dataset (DVC tracked)
│   │   └── emi_prediction_dataset.csv.dvc
│   ├── Interim/                           # Intermediate processed data
│   └── Processed/                         # Train/validation/test splits
├── artifacts/
│   ├── best_params/                       # Best model parameters
│   ├── models/                            # Trained model artifacts
│   └── preprocessor/                      # Preprocessor artifacts
├── Notebooks/
│   └── emi.ipynb                          # Exploratory data analysis
├── src/
│   ├── components/                        # Data processing components
│   │   ├── data_ingestion.py
│   │   ├── data_cleaning.py
│   │   ├── feature_engineering.py
│   │   ├── reg_trainer.py
│   │   ├── cls_trainer.py
│   │   └── final_trainer.py
│   ├── utils.py                           # Utility functions
│   ├── logger.py                          # Logging configuration
│   ├── config.py                          # Project configurations
│   ├── mlflow_config.py                   # MLflow configuration
│   └── exception.py                       # Custom exception handling
├── pages/
│   ├── 1_Predict_EMI.py                   # Prediction interface
│   ├── 2_Data_Explorer.py                 # Data exploration page
│   └── 3_Model_Performance.py             # Model metrics dashboard
├── Home.py                                # Main application entry point
├── requirements.txt                       # Python dependencies
├── .env.example                           # Environment variables template
└── README.md                              # Project documentation
```

---

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- Git
- DagsHub account (for MLflow tracking)

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Pooja-Spandana/EMI_Prediction.git
   cd EMI_Prediction
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate.bat
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   # DagsHub Authentication
   DAGSHUB_USERNAME=your_username
   DAGSHUB_REPO_NAME=EMI_Prediction
   DAGSHUB_TOKEN=your_dagshub_token
   
   # MLflow Authentication
   MLFLOW_TRACKING_USERNAME=your_username
   MLFLOW_TRACKING_PASSWORD=your_dagshub_token
   MLFLOW_TRACKING_URI=https://dagshub.com/your_username/EMI_Prediction.mlflow
   
   # Streamlit Cloud flag
   STREAMLIT_CLOUD=false
   ```

5. **Pull DVC data**
   ```bash
   dvc pull
   ```

6. **Run the application**
   ```bash
   streamlit run Home.py
   ```

The app will open in your browser at `http://localhost:8501`

---

## ☁️ Deployment

### Streamlit Cloud Deployment

1. **Push code to GitHub**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click **"New app"**
   - Select your repository
   - Set main file: `Home.py`
   - Click **"Deploy"**

3. **Add Secrets**
   
   In Streamlit Cloud Settings → Secrets, add:
   ```toml
   DAGSHUB_USERNAME = "your_username"
   DAGSHUB_REPO_NAME = "EMI_Prediction"
   DAGSHUB_TOKEN = "your_token"
   MLFLOW_TRACKING_USERNAME = "your_username"
   MLFLOW_TRACKING_PASSWORD = "your_token"
   MLFLOW_TRACKING_URI = "https://dagshub.com/your_username/EMI_Prediction.mlflow"
   STREAMLIT_CLOUD = "true"
   ```

---

## 🔬 MLflow & DagsHub Integration

### Experiment Tracking

All training runs are logged to MLflow with:
- **Parameters**: Model hyperparameters, preprocessing settings
- **Metrics**: Accuracy, precision, recall, F1, R², RMSE, MAE, MAPE
- **Artifacts**: Trained models, confusion matrices, feature importance plots

### Model Registry

Production models are registered in the MLflow Model Registry:
- **Classification Model**: `emi_eligibility_classifier` (LightGBM)
- **Regression Model**: `emi_max_monthly_predictor` (XGBoost)

### Accessing MLflow UI

Visit your DagsHub repository's MLflow tab:
```
https://dagshub.com/Pooja-Spandana/EMI_Prediction.mlflow
```

---

## 📦 DVC Data Versioning

### Data Pipeline

The project uses DVC to version control the raw dataset:

1. **Track data**
   ```bash
   dvc add Data/Raw/emi_prediction_dataset.csv
   ```

2. **Push to remote**
   ```bash
   dvc push
   ```

3. **Pull data (for new users)**
   ```bash
   dvc pull
   ```

### DVC Remote

Data is stored on DagsHub's S3-compatible storage:
```
https://dagshub.com/Pooja-Spandana/EMI_Prediction/raw/main/Data/Raw/emi_prediction_dataset.csv
```

---

## 👥 Authors

**Pooja Spandana**
- GitHub: [@Pooja-Spandana](https://github.com/Pooja-Spandana)
- DagsHub: [Pooja-Spandana](https://dagshub.com/Pooja-Spandana)

---

## 🙏 Acknowledgments

- **MLflow** for experiment tracking and model management
- **DagsHub** for providing remote MLflow tracking and DVC storage
- **Streamlit** for the amazing web framework
- **XGBoost** and **LightGBM** teams for powerful ML libraries

---

<div align="center">
  <p><strong>EMI Prediction Platform v1.0</strong></p>
</div>