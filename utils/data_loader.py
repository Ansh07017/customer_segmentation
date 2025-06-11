import pandas as pd
import streamlit as st
import numpy as np

def load_data(uploaded_file):
    """
    Load and validate the customer segmentation dataset.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        pandas.DataFrame: Cleaned and validated dataset
    """
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Validate required columns
        required_columns = ['CustomerID', 'Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            # Try to find similar column names
            column_mapping = {}
            for req_col in missing_columns:
                for df_col in df.columns:
                    if any(keyword in df_col.lower() for keyword in get_column_keywords(req_col)):
                        column_mapping[df_col] = req_col
                        break
            
            # Rename columns if mapping found
            if column_mapping:
                df = df.rename(columns=column_mapping)
                st.info(f"Renamed columns: {column_mapping}")
            else:
                raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Data validation and cleaning
        df = clean_data(df)
        
        return df
        
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def get_column_keywords(column_name):
    """
    Get keywords to match column names.
    
    Args:
        column_name (str): The required column name
        
    Returns:
        list: List of keywords to match
    """
    keyword_mapping = {
        'CustomerID': ['customer', 'id', 'customerid'],
        'Genre': ['gender', 'genre', 'sex'],
        'Age': ['age'],
        'Annual Income (k$)': ['income', 'annual', 'salary'],
        'Spending Score (1-100)': ['spending', 'score', 'spend']
    }
    
    return keyword_mapping.get(column_name, [])

def clean_data(df):
    """
    Clean and validate the dataset.
    
    Args:
        df (pandas.DataFrame): Raw dataset
        
    Returns:
        pandas.DataFrame: Cleaned dataset
    """
    # Make a copy to avoid modifying original
    df_clean = df.copy()
    
    # Remove duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    
    if len(df_clean) < initial_rows:
        st.info(f"Removed {initial_rows - len(df_clean)} duplicate rows")
    
    # Handle missing values
    missing_values = df_clean.isnull().sum()
    if missing_values.any():
        st.warning("Found missing values in the dataset:")
        st.write(missing_values[missing_values > 0])
        
        # Fill missing values with appropriate strategies
        if 'Age' in df_clean.columns:
            df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
        
        if 'Annual Income (k$)' in df_clean.columns:
            df_clean['Annual Income (k$)'].fillna(df_clean['Annual Income (k$)'].median(), inplace=True)
        
        if 'Spending Score (1-100)' in df_clean.columns:
            df_clean['Spending Score (1-100)'].fillna(df_clean['Spending Score (1-100)'].median(), inplace=True)
        
        if 'Genre' in df_clean.columns:
            df_clean['Genre'].fillna(df_clean['Genre'].mode()[0], inplace=True)
    
    # Validate data ranges
    if 'Age' in df_clean.columns:
        df_clean = df_clean[(df_clean['Age'] >= 0) & (df_clean['Age'] <= 120)]
    
    if 'Annual Income (k$)' in df_clean.columns:
        df_clean = df_clean[df_clean['Annual Income (k$)'] >= 0]
    
    if 'Spending Score (1-100)' in df_clean.columns:
        df_clean = df_clean[(df_clean['Spending Score (1-100)'] >= 1) & (df_clean['Spending Score (1-100)'] <= 100)]
    
    # Standardize Genre values
    if 'Genre' in df_clean.columns:
        df_clean['Genre'] = df_clean['Genre'].str.title()
        df_clean['Genre'] = df_clean['Genre'].replace({'M': 'Male', 'F': 'Female'})
    
    return df_clean

def get_sample_data():
    """
    Load the actual Mall Customer Segmentation dataset.
    
    Returns:
        pandas.DataFrame: Mall customer segmentation dataset
    """
    try:
        # Load the actual dataset
        df = pd.read_csv('extracted_data/Mall_Customers.csv')
        
        # Standardize column names to match expected format
        column_mapping = {
            'Gender': 'Genre'  # Map Gender to Genre for consistency
        }
        df = df.rename(columns=column_mapping)
        
        # Clean and validate data
        df = clean_data(df)
        
        return df
    except FileNotFoundError:
        # Fallback: create minimal realistic data if file not found
        df = pd.DataFrame({
            'CustomerID': range(1, 11),
            'Genre': ['Male', 'Female'] * 5,
            'Age': [25, 30, 35, 40, 45, 28, 32, 38, 42, 48],
            'Annual Income (k$)': [50, 60, 70, 80, 90, 55, 65, 75, 85, 95],
            'Spending Score (1-100)': [50, 60, 40, 70, 30, 80, 45, 85, 35, 75]
        })
        return df

def validate_data_quality(df):
    """
    Validate the quality of the loaded data.
    
    Args:
        df (pandas.DataFrame): Dataset to validate
        
    Returns:
        dict: Data quality report
    """
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    # Check for outliers in numerical columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    outliers = {}
    
    for col in numerical_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_count = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
        outliers[col] = outlier_count
    
    report['outliers'] = outliers
    
    return report
