# Patient Clustering App

## Project Overview
This project aims to cluster patients based on their clinical and demographic features using unsupervised machine learning algorithms. The goal is to identify meaningful patient groups that can help in understanding patient characteristics, potential risk factors, and medical intervention needs. The app allows users to input patient details and predicts cluster assignment using **KMeans** and **DBSCAN**.

**Dataset link**:https://www.kaggle.com/datasets/arjunnsharma/patient-dataset-for-clustering-raw-data

**Streamlit link**:https://kanika-07cs-task17-16-10-25--app-dfkvvm.streamlit.app/

## Dataset Description
The dataset contains patient information with 16 features:

| Feature | Description |
|---------|-------------|
| age | Patient age (years) |
| gender | Patient gender (Male/Female) |
| chest_pain_type | Type of chest pain (Type1–Type4) |
| blood_pressure | Resting blood pressure (mmHg) |
| cholesterol | Serum cholesterol (mg/dL) |
| max_heart_rate | Maximum heart rate achieved |
| exercise_angina | Exercise-induced angina (Yes/No) |
| plasma_glucose | Plasma glucose level |
| skin_thickness | Triceps skinfold thickness (mm) |
| insulin | Insulin level |
| bmi | Body Mass Index (kg/m²) |
| diabetes_pedigree | Diabetes pedigree function |
| hypertension | Presence of hypertension (Yes/No) |
| heart_disease | Ground truth label for heart disease (0/1) |
| residence_type | Residence type (Urban/Rural) |
| smoking_status | Smoking status (Smoker/Non-Smoker) |

The **heart_disease** column is used as the ground truth for evaluati

## Data Preprocessing Steps
1. **Handling Missing Values**:
   - Numeric columns: Filled missing values with mean.
   - Categorical columns: Filled missing values with mode.

2. **Encoding Categorical Features**:
   - Binary/categorical columns encoded using LabelEncoder.
   - Ensured consistent column order for model input.

3. **Outlier Treatment**:
   - Applied IQR-based capping to numerical columns to handle extreme values.

4. **Feature Scaling**:
   - Applied StandardScaler to normalize numerical featur

## Model Development
### KMeans Clustering
- Trained on scaled features.
- Optimal number of clusters determined using Elbow Method.

### DBSCAN Clustering
- Density-based clustering algorithm to detect clusters of varying shapes and sizes.
- Parameters determined using k-distance graph.
- Pre-trained DBSCAN model saved for prediction.

## Evaluation
Metrics used to evaluate clustering:

1. **Silhouette Score**: Measures how similar a point is to its cluster vs. other clusters.  
2. **Adjusted Rand Index (ARI)**: Measures similarity to the ground truth (heart_disease).


## Results & Insights
- **KMeans clusters** capture patient groups based on similarity in numeric features (e.g., age, BMI, cholesterol).  
- **DBSCAN clusters** effectively identify dense regions and mark noise points (-1) for outliers.  
- Cluster profiles can be analyzed by calculating mean feature values per cluster, helping clinicians identify high-risk patient groups or those needing targeted interventions.

## How to Run
1. Clone the repository:
   - git clone <repository_url>
   - cd <repo_url>
2. Run:
   - streamlit run app.py

## Conclusion
This project demonstrates patient segmentation using unsupervised learning. By leveraging KMeans and DBSCAN, we can identify groups of patients with similar clinical characteristics. While DBSCAN 
effectively identifies dense clusters and noise points, KMeans provides a simple and interpretable clustering approach. This clustering can be used as a foundation for further clinical analysis, 
risk stratification, and targeted interventions.
