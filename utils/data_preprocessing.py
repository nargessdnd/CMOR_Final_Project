# utils/data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def handle_missing_values(df):
    """
    Handles missing values in the DataFrame.
    Fills numerical columns with median and categorical columns with mode.
    """
    df_processed = df.copy()
    for col in df_processed.columns:
        if df_processed[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df_processed[col]):
                median_val = df_processed[col].median()
                df_processed[col].fillna(median_val, inplace=True)
                print(f"Filled missing values in numerical column '{col}' with median ({median_val}).")
            elif pd.api.types.is_object_dtype(df_processed[col]):
                mode_val = df_processed[col].mode()[0]
                df_processed[col].fillna(mode_val, inplace=True)
                print(f"Filled missing values in categorical column '{col}' with mode ('{mode_val}').")
    return df_processed

def encode_categorical_features(df):
    """
    Encodes categorical features using Label Encoding.
    """
    df_processed = df.copy()
    label_encoders = {}
    for col in df_processed.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
        print(f"Label encoded categorical column '{col}'.")
    return df_processed, label_encoders

def scale_numerical_features(df):
    """
    Scales numerical features using StandardScaler.
    """
    df_processed = df.copy()
    numerical_cols = df_processed.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
    print(f"Scaled numerical columns: {', '.join(numerical_cols)}.")
    return df_processed, scaler

def handle_outliers_iqr(df, columns=None, factor=1.5):
    """
    Handles outliers using the IQR method. Replaces outliers with NaN,
    which can then be imputed or handled otherwise.
    Alternatively, capping can be implemented. Here we just identify.
    """
    df_processed = df.copy()
    if columns is None:
        columns = df_processed.select_dtypes(include=np.number).columns

    outlier_indices = []
    for col in columns:
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        # Find outliers
        col_outlier_indices = df_processed[(df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)].index
        outlier_indices.extend(col_outlier_indices)

    # Remove duplicates
    outlier_indices = list(set(outlier_indices))
    print(f"Identified {len(outlier_indices)} rows with outliers based on IQR method in columns: {', '.join(columns)}.")

    # Optional: Remove outliers (or cap them)
    # df_processed = df_processed.drop(outlier_indices)
    # print(f"Removed {len(outlier_indices)} rows containing outliers.")

    # For now, just return the dataframe without removing them, but print the info.
    # Imputation in handle_missing_values might mitigate some effect if outliers were replaced by NaN first.
    return df_processed


def preprocess_esg_data(df, handle_outliers=True):
    """
    Applies a full preprocessing pipeline to the ESG dataset.
    1. Handle missing values
    2. Encode categorical features
    3. Optional: Handle outliers (by identification, not removal here)
    4. Scale numerical features
    """
    print("Starting preprocessing...")
    # 1. Handle missing values
    df_processed = handle_missing_values(df)

    # 2. Encode categorical features
    df_processed, _ = encode_categorical_features(df_processed) # Ignore label_encoders for now

    # 3. Optional: Handle outliers
    if handle_outliers:
        # Only apply outlier detection to original numerical columns before scaling
        numerical_cols_original = df.select_dtypes(include=np.number).columns
        # Identify outliers (doesn't remove them in this implementation)
        df_processed = handle_outliers_iqr(df_processed, columns=[col for col in numerical_cols_original if col in df_processed.columns])
        # Re-impute if outliers were replaced by NaN (not done here)
        # df_processed = handle_missing_values(df_processed)


    # 4. Scale numerical features (all columns are now numeric)
    df_scaled, _ = scale_numerical_features(df_processed) # Ignore scaler for now

    print("Preprocessing finished.")
    return df_scaled


def create_feature_target_split(df, target_column):
    """
    Splits the DataFrame into features (X) and target (y).
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    print(f"Created feature matrix X (shape: {X.shape}) and target vector y (shape: {y.shape}) with target '{target_column}'.")
    return X, y