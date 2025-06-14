import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

def perform_eda(df):
    """
    Perform comprehensive Exploratory Data Analysis.
    
    Args:
        df (pandas.DataFrame): The dataset to analyze
        
    Returns:
        dict: Dictionary containing all EDA visualizations
    """
    eda_results = {}
    
    # Age distribution
    eda_results['age_dist'] = create_age_distribution(df)
    
    # Income distribution
    eda_results['income_dist'] = create_income_distribution(df)
    
    # Spending score distribution
    eda_results['spending_dist'] = create_spending_distribution(df)
    
    # Gender distribution
    eda_results['gender_dist'] = create_gender_distribution(df)
    
    # Gender vs Income
    eda_results['gender_income'] = create_gender_income_analysis(df)
    
    # Gender vs Spending
    eda_results['gender_spending'] = create_gender_spending_analysis(df)
    
    # Age vs Income scatter
    eda_results['age_income_scatter'] = create_age_income_scatter(df)
    
    # Age vs Spending scatter
    eda_results['age_spending_scatter'] = create_age_spending_scatter(df)
    
    return eda_results

def create_age_distribution(df):
    """Create age distribution histogram."""
    fig = px.histogram(
        df, 
        x='Age',
        nbins=20,
        title='Age Distribution of Customers',
        labels={'Age': 'Age (years)', 'count': 'Number of Customers'},
        color_discrete_sequence=['#FF6B6B']
    )
    
    # Add mean line
    mean_age = df['Age'].mean()
    fig.add_vline(
        x=mean_age, 
        line_dash="dash", 
        line_color="#FFEAA7",
        annotation_text=f"Mean: {mean_age:.1f}",
        annotation_position="top"
    )
    
    fig.update_layout(
        showlegend=False,
        height=380,
        plot_bgcolor='rgba(26, 22, 37, 0.8)',
        paper_bgcolor='rgba(26, 22, 37, 0.8)',
        font=dict(color='#f3f4f6', size=11),
        title=dict(font=dict(color='#ffffff', size=15, family='Arial Black')),
        xaxis=dict(
            gridcolor='rgba(111, 66, 193, 0.2)', 
            color='#ffffff',
            title_font=dict(color='#ffffff', size=12)
        ),
        yaxis=dict(
            gridcolor='rgba(111, 66, 193, 0.2)', 
            color='#ffffff',
            title_font=dict(color='#ffffff', size=12)
        ),
        margin=dict(l=50, r=50, t=60, b=50)
    )
    
    return fig

def create_income_distribution(df):
    """Create income distribution histogram."""
    fig = px.histogram(
        df, 
        x='Annual Income (k$)',
        nbins=20,
        title='Annual Income Distribution',
        labels={'Annual Income (k$)': 'Annual Income (k$)', 'count': 'Number of Customers'},
        color_discrete_sequence=['#4ECDC4']
    )
    
    # Add mean line
    mean_income = df['Annual Income (k$)'].mean()
    fig.add_vline(
        x=mean_income, 
        line_dash="dash", 
        line_color="#FFEAA7",
        annotation_text=f"Mean: ${mean_income:.1f}k",
        annotation_position="top"
    )
    
    fig.update_layout(
        showlegend=False,
        height=380,
        plot_bgcolor='rgba(26, 22, 37, 0.8)',
        paper_bgcolor='rgba(26, 22, 37, 0.8)',
        font=dict(color='#f3f4f6', size=11),
        title=dict(font=dict(color='#ffffff', size=15, family='Arial Black')),
        xaxis=dict(
            gridcolor='rgba(111, 66, 193, 0.2)', 
            color='#ffffff',
            title_font=dict(color='#ffffff', size=12)
        ),
        yaxis=dict(
            gridcolor='rgba(111, 66, 193, 0.2)', 
            color='#ffffff',
            title_font=dict(color='#ffffff', size=12)
        ),
        margin=dict(l=50, r=50, t=60, b=50)
    )
    
    return fig

def create_spending_distribution(df):
    """Create spending score distribution histogram."""
    fig = px.histogram(
        df, 
        x='Spending Score (1-100)',
        nbins=20,
        title='Spending Score Distribution',
        labels={'Spending Score (1-100)': 'Spending Score', 'count': 'Number of Customers'},
        color_discrete_sequence=['#45B7D1']
    )
    
    # Add mean line
    mean_spending = df['Spending Score (1-100)'].mean()
    fig.add_vline(
        x=mean_spending, 
        line_dash="dash", 
        line_color="#FFEAA7",
        annotation_text=f"Mean: {mean_spending:.1f}",
        annotation_position="top"
    )
    
    fig.update_layout(
        showlegend=False,
        height=380,
        plot_bgcolor='rgba(26, 22, 37, 0.8)',
        paper_bgcolor='rgba(26, 22, 37, 0.8)',
        font=dict(color='#f3f4f6', size=11),
        title=dict(font=dict(color='#ffffff', size=15, family='Arial Black')),
        xaxis=dict(
            gridcolor='rgba(111, 66, 193, 0.2)', 
            color='#ffffff',
            title_font=dict(color='#ffffff', size=12)
        ),
        yaxis=dict(
            gridcolor='rgba(111, 66, 193, 0.2)', 
            color='#ffffff',
            title_font=dict(color='#ffffff', size=12)
        ),
        margin=dict(l=50, r=50, t=60, b=50)
    )
    
    return fig

def create_gender_distribution(df):
    """Create gender distribution pie chart."""
    gender_counts = df['Genre'].value_counts()
    
    fig = px.pie(
        values=gender_counts.values,
        names=gender_counts.index,
        title='Gender Distribution',
        color_discrete_sequence=['#FF6B6B', '#4ECDC4']
    )
    
    fig.update_layout(
        height=550,
        plot_bgcolor='rgba(26, 22, 37, 0.8)',
        paper_bgcolor='rgba(26, 22, 37, 0.8)',
        font=dict(color='#f3f4f6', size=12),
        title=dict(font=dict(color='#ffffff', size=16, family='Arial Black')),
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    return fig

def create_gender_income_analysis(df):
    """Create gender vs income box plot."""
    fig = px.box(
        df, 
        x='Genre', 
        y='Annual Income (k$)',
        title='Income Distribution by Gender',
        color='Genre',
        color_discrete_sequence=['#FF6B6B', '#4ECDC4']
    )
    
    fig.update_layout(
        showlegend=False,
        height=380,
        plot_bgcolor='rgba(26, 22, 37, 0.8)',
        paper_bgcolor='rgba(26, 22, 37, 0.8)',
        font=dict(color='#f3f4f6', size=11),
        title=dict(font=dict(color='#ffffff', size=15, family='Arial Black')),
        xaxis=dict(
            gridcolor='rgba(111, 66, 193, 0.2)', 
            color='#ffffff',
            title_font=dict(color='#ffffff', size=12)
        ),
        yaxis=dict(
            gridcolor='rgba(111, 66, 193, 0.2)', 
            color='#ffffff',
            title_font=dict(color='#ffffff', size=12)
        ),
        margin=dict(l=50, r=50, t=60, b=50)
    )
    
    return fig

def create_gender_spending_analysis(df):
    """Create gender vs spending box plot."""
    fig = px.box(
        df, 
        x='Genre', 
        y='Spending Score (1-100)',
        title='Spending Score Distribution by Gender',
        color='Genre',
        color_discrete_sequence=['#45B7D1', '#96CEB4']
    )
    
    fig.update_layout(
        showlegend=False,
        height=380,
        plot_bgcolor='rgba(26, 22, 37, 0.8)',
        paper_bgcolor='rgba(26, 22, 37, 0.8)',
        font=dict(color='#f3f4f6', size=11),
        title=dict(font=dict(color='#ffffff', size=15, family='Arial Black')),
        xaxis=dict(
            gridcolor='rgba(111, 66, 193, 0.2)', 
            color='#ffffff',
            title_font=dict(color='#ffffff', size=12)
        ),
        yaxis=dict(
            gridcolor='rgba(111, 66, 193, 0.2)', 
            color='#ffffff',
            title_font=dict(color='#ffffff', size=12)
        ),
        margin=dict(l=50, r=50, t=60, b=50)
    )
    
    return fig

def create_age_income_scatter(df):
    """Create age vs income scatter plot."""
    fig = px.scatter(
        df, 
        x='Age', 
        y='Annual Income (k$)',
        color='Genre',
        title='Age vs Annual Income',
        hover_data=['Spending Score (1-100)'],
        color_discrete_sequence=['#FFEAA7', '#DDA0DD']
    )
    
    fig.update_layout(
        height=380,
        plot_bgcolor='rgba(26, 22, 37, 0.8)',
        paper_bgcolor='rgba(26, 22, 37, 0.8)',
        font=dict(color='#f3f4f6', size=11),
        title=dict(font=dict(color='#ffffff', size=15, family='Arial Black')),
        xaxis=dict(
            gridcolor='rgba(111, 66, 193, 0.2)', 
            color='#ffffff',
            title_font=dict(color='#ffffff', size=12)
        ),
        yaxis=dict(
            gridcolor='rgba(111, 66, 193, 0.2)', 
            color='#ffffff',
            title_font=dict(color='#ffffff', size=12)
        ),
        legend=dict(
            font=dict(color='#ffffff', size=10), 
            bgcolor='rgba(42, 31, 61, 0.9)',
            bordercolor='rgba(111, 66, 193, 0.3)',
            borderwidth=1
        ),
        margin=dict(l=50, r=50, t=60, b=50)
    )
    
    return fig

def create_age_spending_scatter(df):
    """Create age vs spending scatter plot."""
    fig = px.scatter(
        df, 
        x='Age', 
        y='Spending Score (1-100)',
        color='Genre',
        title='Age vs Spending Score',
        hover_data=['Annual Income (k$)'],
        color_discrete_sequence=['#F0A500', '#74B9FF']
    )
    
    fig.update_layout(
        height=380,
        plot_bgcolor='rgba(26, 22, 37, 0.8)',
        paper_bgcolor='rgba(26, 22, 37, 0.8)',
        font=dict(color='#f3f4f6', size=11),
        title=dict(font=dict(color='#ffffff', size=15, family='Arial Black')),
        xaxis=dict(
            gridcolor='rgba(111, 66, 193, 0.2)', 
            color='#ffffff',
            title_font=dict(color='#ffffff', size=12)
        ),
        yaxis=dict(
            gridcolor='rgba(111, 66, 193, 0.2)', 
            color='#ffffff',
            title_font=dict(color='#ffffff', size=12)
        ),
        legend=dict(
            font=dict(color='#ffffff', size=10), 
            bgcolor='rgba(42, 31, 61, 0.9)',
            bordercolor='rgba(111, 66, 193, 0.3)',
            borderwidth=1
        ),
        margin=dict(l=50, r=50, t=60, b=50)
    )
    
    return fig

def show_data_info(df):
    """Display comprehensive data information."""
    st.subheader("Dataset Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Shape:**")
        st.write(f"- Rows: {df.shape[0]}")
        st.write(f"- Columns: {df.shape[1]}")
        
        st.write("**Data Types:**")
        for col, dtype in df.dtypes.items():
            st.write(f"- {col}: {dtype}")
    
    with col2:
        st.write("**Missing Values:**")
        missing_values = df.isnull().sum()
        if missing_values.any():
            for col, count in missing_values.items():
                if count > 0:
                    st.write(f"- {col}: {count}")
        else:
            st.write("No missing values found ✅")
        
        st.write("**Unique Values:**")
        for col in df.columns:
            unique_count = df[col].nunique()
            st.write(f"- {col}: {unique_count}")

def get_statistical_summary(df):
    """Get comprehensive statistical summary."""
    summary = {}
    
    # Numerical columns summary
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    summary['numerical'] = df[numerical_cols].describe()
    
    # Categorical columns summary
    categorical_cols = df.select_dtypes(include=['object']).columns
    summary['categorical'] = {}
    
    for col in categorical_cols:
        summary['categorical'][col] = {
            'unique_values': df[col].nunique(),
            'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
            'value_counts': df[col].value_counts().to_dict()
        }
    
    return summary

def detect_outliers(df):
    """Detect outliers using IQR method."""
    outliers = {}
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = outlier_mask.sum()
        
        outliers[col] = {
            'count': outlier_count,
            'percentage': (outlier_count / len(df)) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_values': df.loc[outlier_mask, col].tolist()
        }
    
    return outliers

def create_feature_relationships(df):
    """Create pairwise relationships between features."""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) >= 2:
        fig = px.scatter_matrix(
            df, 
            dimensions=numerical_cols,
            color='Genre' if 'Genre' in df.columns else None,
            title='Feature Relationships Matrix'
        )
        
        fig.update_traces(diagonal_visible=False)
        fig.update_layout(height=600)
        
        return fig
    
    return None

def analyze_customer_segments_preview(df):
    """Provide a preview of potential customer segments based on EDA."""
    insights = []
    
    # Age-based insights
    age_median = df['Age'].median()
    young_customers = len(df[df['Age'] < age_median])
    older_customers = len(df[df['Age'] >= age_median])
    
    insights.append(f"**Age Distribution:** {young_customers} customers under {age_median:.0f} years, {older_customers} customers {age_median:.0f} years and above")
    
    # Income-based insights
    income_median = df['Annual Income (k$)'].median()
    low_income = len(df[df['Annual Income (k$)'] < income_median])
    high_income = len(df[df['Annual Income (k$)'] >= income_median])
    
    insights.append(f"**Income Distribution:** {low_income} customers with income below ${income_median:.0f}k, {high_income} customers with income ${income_median:.0f}k and above")
    
    # Spending-based insights
    spending_median = df['Spending Score (1-100)'].median()
    low_spenders = len(df[df['Spending Score (1-100)'] < spending_median])
    high_spenders = len(df[df['Spending Score (1-100)'] >= spending_median])
    
    insights.append(f"**Spending Behavior:** {low_spenders} customers with spending score below {spending_median:.0f}, {high_spenders} customers with spending score {spending_median:.0f} and above")
    
    # Gender-based insights
    if 'Genre' in df.columns:
        gender_counts = df['Genre'].value_counts()
        insights.append(f"**Gender Distribution:** {gender_counts.to_dict()}")
    
    return insights
