### preprocess.py ###

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

def preprocess_data(file):
    # Detect file type and load
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext in ['.xls', '.xlsx']:
        data = pd.read_excel(file)
    elif ext == '.csv':
        data = pd.read_csv(file)
    else:
        raise ValueError("Unsupported file format. Please upload .csv, .xls, or .xlsx.")

    if 'EmpNumber' not in data.columns:
        raise ValueError("Input file must contain 'EmpNumber' column.")

    emp_numbers = data['EmpNumber'].copy()

    # Handle missing values
    imputer = SimpleImputer(strategy='most_frequent')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    data_imputed = data_imputed.convert_dtypes()

    def safe_map(column, mapping, default=-1):
        if column in data_imputed.columns:
            data_imputed[column] = data_imputed[column].map(mapping).fillna(default)

    safe_map('Gender', {'Male': 1, 'Female': 0})
    safe_map('EducationBackground', {
        'Life Sciences': 5, 'Medical': 4, 'Marketing': 3,
        'Technical Degree': 2, 'Other': 1, 'Human Resources': 0
    })
    safe_map('MaritalStatus', {'Married': 2, 'Single': 1, 'Divorced': 0})
    safe_map('EmpDepartment', {
        'Sales': 5, 'Development': 4, 'Research & Development': 3,
        'Human Resources': 2, 'Finance': 1, 'Data Science': 0
    })
    safe_map('EmpJobRole', {
        'Sales Executive': 18, 'Developer': 17, 'Manager R&D': 16,
        'Research Scientist': 15, 'Sales Representative': 14,
        'Laboratory Technician': 13, 'Senior Developer': 12,
        'Manager': 11, 'Finance Manager': 10, 'Human Resources': 9,
        'Technical Lead': 8, 'Manufacturing Director': 7,
        'Healthcare Representative': 6, 'Data Scientist': 5,
        'Research Director': 4, 'Business Analyst': 3,
        'Senior Manager R&D': 2, 'Delivery Manager': 1,
        'Technical Architect': 0
    })
    safe_map('BusinessTravelFrequency', {
        'Travel_Rarely': 2, 'Travel_Frequently': 1, 'Non-Travel': 0
    })
    safe_map('OverTime', {'No': 1, 'Yes': 0})
    safe_map('Attrition', {'No': 1, 'Yes': 0})

    selected_features = [
        'Age', 'Gender', 'EducationBackground', 'MaritalStatus',
        'EmpDepartment', 'EmpJobRole', 'BusinessTravelFrequency',
        'OverTime', 'DistanceFromHome', 'EmpHourlyRate',
        'EmpLastSalaryHikePercent', 'TotalWorkExperienceInYears',
        'TrainingTimesLastYear', 'ExperienceYearsAtThisCompany',
        'ExperienceYearsInCurrentRole', 'YearsSinceLastPromotion',
        'YearsWithCurrManager', 'Attrition', 'PerformanceRating',
        'NumCompaniesWorked', 'EnvironmentSatisfaction',
        'JobSatisfaction', 'WorkLifeBalance', 'JobInvolvement',
        'PercentSalaryHike'
    ]

    default_values = {
        'EnvironmentSatisfaction': 3,
        'JobSatisfaction': 3,
        'WorkLifeBalance': 3,
        'JobInvolvement': 3,
        'PercentSalaryHike': 10,
        'PerformanceRating': 3
    }

    for col in selected_features:
        if col not in data_imputed.columns:
            data_imputed[col] = default_values.get(col, 0)

    def handle_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        min_limit = Q1 - 1.5 * IQR
        max_limit = Q3 + 1.5 * IQR
        df[column] = np.where(df[column] > max_limit, df[column].median(), df[column])
        df[column] = np.where(df[column] < min_limit, df[column].median(), df[column])

    for col in selected_features:
        handle_outliers(data_imputed, col)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(data_imputed[selected_features])
    final_df = pd.DataFrame(features_scaled, columns=selected_features)

    return final_df, emp_numbers, data_imputed
