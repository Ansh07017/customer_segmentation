import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import utility functions
from utils.data_loader import load_data, get_sample_data
from utils.eda_functions import perform_eda, show_data_info
from utils.clustering import perform_clustering, get_cluster_insights
from utils.visualizations import create_cluster_visualization, create_correlation_heatmap

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🛍️ Customer Segmentation Dashboard")
    st.markdown("### Using K-Means Clustering for Mall Customer Analysis")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Data Overview", "Exploratory Data Analysis", "Customer Segmentation", "Cluster Analysis"]
    )
    
    # Data loading section
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Loading")
    
    # Option to upload file or use sample data
    data_option = st.sidebar.radio(
        "Choose data source:",
        ["Upload CSV File", "Use Sample Data"]
    )
    
    df = None
    
    if data_option == "Upload CSV File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Mall Customer Segmentation CSV",
            type=['csv'],
            help="Upload the Mall Customer Segmentation dataset from Kaggle"
        )
        
        if uploaded_file is not None:
            try:
                df = load_data(uploaded_file)
                st.sidebar.success("✅ Data loaded successfully!")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading data: {str(e)}")
                return
        else:
            st.sidebar.info("Please upload a CSV file to continue")
            st.info("👆 Please upload the Mall Customer Segmentation dataset using the sidebar to get started.")
            st.markdown("""
            **Expected dataset format:**
            - CustomerID: Unique identifier for each customer
            - Genre: Customer gender (Male/Female)
            - Age: Customer age
            - Annual Income (k$): Annual income in thousands of dollars
            - Spending Score (1-100): Score assigned based on customer behavior and spending nature
            """)
            return
    else:
        # Use sample data
        df = get_sample_data()
        st.sidebar.success("✅ Sample data loaded!")
        st.sidebar.info("Using sample Mall Customer Segmentation data for demonstration")
    
    # Main content based on selected page
    if page == "Data Overview":
        show_data_overview(df)
    elif page == "Exploratory Data Analysis":
        show_eda(df)
    elif page == "Customer Segmentation":
        show_clustering(df)
    elif page == "Cluster Analysis":
        show_cluster_analysis(df)

def show_data_overview(df):
    st.header("📊 Data Overview")
    
    # Display basic information
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", len(df))
    
    with col2:
        st.metric("Features", len(df.columns))
    
    with col3:
        st.metric("Male Customers", len(df[df['Genre'] == 'Male']))
    
    with col4:
        st.metric("Female Customers", len(df[df['Genre'] == 'Female']))
    
    # Show dataset info
    show_data_info(df)
    
    # Display first few rows
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Basic statistics
    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

def show_eda(df):
    st.header("🔍 Exploratory Data Analysis")
    
    # Perform EDA
    eda_results = perform_eda(df)
    
    # Display EDA results
    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Correlations", "Gender Analysis", "Age Groups"])
    
    with tab1:
        st.subheader("Feature Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(eda_results['age_dist'], use_container_width=True)
            st.plotly_chart(eda_results['income_dist'], use_container_width=True)
        
        with col2:
            st.plotly_chart(eda_results['spending_dist'], use_container_width=True)
            st.plotly_chart(eda_results['gender_dist'], use_container_width=True)
    
    with tab2:
        st.subheader("Correlation Analysis")
        st.plotly_chart(create_correlation_heatmap(df), use_container_width=True)
        
        st.markdown("""
        **Key Insights from Correlation Analysis:**
        - Analyze the relationships between numerical features
        - Identify which features are most correlated
        - Understand potential clustering patterns
        """)
    
    with tab3:
        st.subheader("Gender-based Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(eda_results['gender_income'], use_container_width=True)
        
        with col2:
            st.plotly_chart(eda_results['gender_spending'], use_container_width=True)
    
    with tab4:
        st.subheader("Age Group Analysis")
        st.plotly_chart(eda_results['age_income_scatter'], use_container_width=True)
        st.plotly_chart(eda_results['age_spending_scatter'], use_container_width=True)

def show_clustering(df):
    st.header("🎯 Customer Segmentation")
    
    # Clustering parameters
    st.subheader("Clustering Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=5)
        features_to_use = st.multiselect(
            "Select Features for Clustering",
            ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'],
            default=['Annual Income (k$)', 'Spending Score (1-100)']
        )
    
    with col2:
        random_state = st.number_input("Random State", min_value=0, max_value=100, value=42)
        scale_features = st.checkbox("Scale Features", value=True)
    
    if len(features_to_use) < 2:
        st.warning("Please select at least 2 features for clustering.")
        return
    
    # Perform clustering
    if st.button("🚀 Run Clustering Analysis"):
        try:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Preparing data for clustering...")
            progress_bar.progress(25)
            
            # Perform clustering
            clustering_results = perform_clustering(
                df, 
                features_to_use, 
                n_clusters, 
                scale_features,
                random_state
            )
            
            progress_bar.progress(75)
            status_text.text("Creating visualizations...")
            
            # Store results in session state
            st.session_state.clustering_results = clustering_results
            st.session_state.features_used = features_to_use
            st.session_state.n_clusters = n_clusters
            
            progress_bar.progress(100)
            status_text.text("✅ Clustering completed!")
            
            # Display results
            display_clustering_results(clustering_results, features_to_use)
            
        except Exception as e:
            st.error(f"Error during clustering: {str(e)}")
    
    # Display previous results if available
    if 'clustering_results' in st.session_state:
        st.markdown("---")
        st.subheader("Previous Clustering Results")
        display_clustering_results(
            st.session_state.clustering_results, 
            st.session_state.features_used
        )

def display_clustering_results(clustering_results, features_used):
    df_clustered = clustering_results['df_clustered']
    centroids = clustering_results['centroids']
    inertia = clustering_results['inertia']
    
    # Display clustering metrics
    st.metric("Within-cluster Sum of Squares (WCSS)", f"{inertia:.2f}")
    
    # Visualization
    st.subheader("Cluster Visualization")
    
    if len(features_used) >= 2:
        fig = create_cluster_visualization(df_clustered, features_used)
        st.plotly_chart(fig, use_container_width=True)
    
    # Cluster distribution
    st.subheader("Cluster Distribution")
    cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = px.pie(
            values=cluster_counts.values,
            names=[f"Cluster {i}" for i in cluster_counts.index],
            title="Customer Distribution by Cluster"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_bar = px.bar(
            x=[f"Cluster {i}" for i in cluster_counts.index],
            y=cluster_counts.values,
            title="Number of Customers per Cluster"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

def show_cluster_analysis(df):
    st.header("📈 Cluster Analysis & Insights")
    
    if 'clustering_results' not in st.session_state:
        st.warning("Please run clustering analysis first in the 'Customer Segmentation' section.")
        return
    
    clustering_results = st.session_state.clustering_results
    features_used = st.session_state.features_used
    n_clusters = st.session_state.n_clusters
    
    df_clustered = clustering_results['df_clustered']
    centroids = clustering_results['centroids']
    
    # Get cluster insights
    insights = get_cluster_insights(df_clustered, features_used)
    
    # Display cluster characteristics
    st.subheader("Cluster Characteristics")
    
    # Create tabs for each cluster
    cluster_tabs = st.tabs([f"Cluster {i}" for i in range(n_clusters)])
    
    for i, tab in enumerate(cluster_tabs):
        with tab:
            cluster_data = df_clustered[df_clustered['Cluster'] == i]
            
            # Basic metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Customers", len(cluster_data))
            
            with col2:
                if 'Age' in features_used:
                    st.metric("Avg Age", f"{cluster_data['Age'].mean():.1f}")
            
            with col3:
                if 'Annual Income (k$)' in features_used:
                    st.metric("Avg Income", f"${cluster_data['Annual Income (k$)'].mean():.1f}k")
            
            with col4:
                if 'Spending Score (1-100)' in features_used:
                    st.metric("Avg Spending Score", f"{cluster_data['Spending Score (1-100)'].mean():.1f}")
            
            # Detailed statistics
            st.subheader(f"Cluster {i} Statistics")
            numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
            st.dataframe(cluster_data[numeric_cols].describe(), use_container_width=True)
            
            # Gender distribution if available
            if 'Genre' in cluster_data.columns:
                st.subheader("Gender Distribution")
                gender_dist = cluster_data['Genre'].value_counts()
                fig_gender = px.pie(
                    values=gender_dist.values,
                    names=gender_dist.index,
                    title=f"Gender Distribution - Cluster {i}"
                )
                st.plotly_chart(fig_gender, use_container_width=True)
    
    # Overall insights
    st.subheader("Business Insights & Recommendations")
    
    # Generate insights based on cluster characteristics
    st.markdown("### Key Findings:")
    
    for cluster_id, insight in insights.items():
        with st.expander(f"Cluster {cluster_id} - {insight['profile']}"):
            st.write(f"**Size:** {insight['size']} customers ({insight['percentage']:.1f}% of total)")
            st.write(f"**Characteristics:** {insight['characteristics']}")
            st.write(f"**Marketing Strategy:** {insight['strategy']}")
    
    # Elbow method for optimal clusters
    st.subheader("Optimal Number of Clusters")
    
    if st.button("🔍 Analyze Optimal Clusters (Elbow Method)"):
        with st.spinner("Analyzing optimal number of clusters..."):
            # Calculate WCSS for different numbers of clusters
            wcss_values = []
            k_range = range(1, 11)
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                
                # Prepare data
                X = df[features_used].copy()
                if st.session_state.get('scale_features', True):
                    scaler = StandardScaler()
                    X = scaler.fit_transform(X)
                
                kmeans.fit(X)
                wcss_values.append(kmeans.inertia_)
            
            # Plot elbow curve
            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(
                x=list(k_range),
                y=wcss_values,
                mode='lines+markers',
                name='WCSS',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ))
            
            fig_elbow.update_layout(
                title='Elbow Method for Optimal Number of Clusters',
                xaxis_title='Number of Clusters (k)',
                yaxis_title='Within-Cluster Sum of Squares (WCSS)',
                showlegend=False
            )
            
            st.plotly_chart(fig_elbow, use_container_width=True)
            
            st.info("The optimal number of clusters is typically at the 'elbow' point where the rate of decrease in WCSS slows down significantly.")

if __name__ == "__main__":
    main()
