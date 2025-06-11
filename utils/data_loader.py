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
    Generate sample Mall Customer Segmentation data for demonstration.
    This function creates realistic sample data when the actual dataset is not available.
    
    Returns:
        pandas.DataFrame: Sample customer segmentation dataset
    """
    np.random.seed(42)
    
    # Generate sample data
    n_customers = 200
    
    # Customer IDs
    customer_ids = range(1, n_customers + 1)
    
    # Generate ages with realistic distribution
    ages = np.random.normal(40, 12, n_customers)
    ages = np.clip(ages, 18, 70).astype(int)
    
    # Generate genders
    genders = np.random.choice(['Male', 'Female'], n_customers, p=[0.44, 0.56])
    
    # Generate income with some correlation to age
    base_income = np.random.normal(50, 20, n_customers)
    age_factor = (ages - 18) * 0.5  # Older people tend to have higher income
    annual_income = base_income + age_factor + np.random.normal(0, 5, n_customers)
    annual_income = np.clip(annual_income, 15, 120).astype(int)
    
    # Generate spending scores with some patterns
    # Create different customer segments
    spending_scores = []
    for i in range(n_customers):
        income = annual_income[i]
        age = ages[i]
        
        # Different spending patterns based on income and age
        if income < 30:  # Low income
            score = np.random.normal(30, 15)
        elif income < 60:  # Medium income
            if age < 35:  # Young medium income - higher spending
                score = np.random.normal(65, 20)
            else:  # Older medium income - moderate spending
                score = np.random.normal(45, 15)
        else:  # High income
            if age < 45:  # Young high income - very high spending
                score = np.random.normal(80, 15)
            else:  # Older high income - moderate to high spending
                score = np.random.normal(60, 20)
        
        spending_scores.append(np.clip(score, 1, 100))
    
    spending_scores = np.array(spending_scores).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'CustomerID': customer_ids,
        'Genre': genders,
        'Age': ages,
        'Annual Income (k$)': annual_income,
        'Spending Score (1-100)': spending_scores
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
