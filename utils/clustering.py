import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import streamlit as st

def perform_clustering(df, features, n_clusters, scale_features=True, random_state=42):
    """
    Perform K-Means clustering on the dataset.
    
    Args:
        df (pandas.DataFrame): The dataset
        features (list): List of features to use for clustering
        n_clusters (int): Number of clusters
        scale_features (bool): Whether to scale features
        random_state (int): Random state for reproducibility
        
    Returns:
        dict: Dictionary containing clustering results
    """
    # Prepare the data
    X = df[features].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Scale features if requested
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values
    
    # Perform K-Means clustering
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init='auto',
        max_iter=300
    )
    
    # Fit the model and get cluster labels
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    
    # Create a copy of the original dataframe with cluster labels
    df_clustered = df.copy()
    df_clustered['Cluster'] = cluster_labels
    
    # Get centroids in original scale
    centroids = kmeans.cluster_centers_
    if scale_features and scaler is not None:
        centroids = scaler.inverse_transform(centroids)
    
    # Create centroids dataframe
    centroids_df = pd.DataFrame(centroids, columns=features)
    centroids_df['Cluster'] = range(n_clusters)
    
    results = {
        'df_clustered': df_clustered,
        'centroids': centroids_df,
        'cluster_labels': cluster_labels,
        'inertia': kmeans.inertia_,
        'silhouette_score': silhouette_avg,
        'scaler': scaler,
        'kmeans_model': kmeans,
        'features_used': features
    }
    
    return results

def get_cluster_insights(df_clustered, features):
    """
    Generate insights for each cluster.
    
    Args:
        df_clustered (pandas.DataFrame): Dataset with cluster labels
        features (list): Features used for clustering
        
    Returns:
        dict: Dictionary containing insights for each cluster
    """
    insights = {}
    n_clusters = df_clustered['Cluster'].nunique()
    total_customers = len(df_clustered)
    
    for cluster_id in range(n_clusters):
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
        cluster_size = len(cluster_data)
        cluster_percentage = (cluster_size / total_customers) * 100
        
        # Calculate cluster characteristics
        characteristics = []
        strategy = []
        age_group = "Unknown"
        income_level = "Unknown"
        spending_level = "Unknown"
        
        # Age analysis
        if 'Age' in features:
            avg_age = cluster_data['Age'].mean()
            if avg_age < 30:
                age_group = "Young"
                characteristics.append(f"Young customers (avg age: {avg_age:.1f})")
                strategy.append("Focus on trendy products and digital marketing")
            elif avg_age < 50:
                age_group = "Middle-aged"
                characteristics.append(f"Middle-aged customers (avg age: {avg_age:.1f})")
                strategy.append("Target family-oriented products and services")
            else:
                age_group = "Mature"
                characteristics.append(f"Mature customers (avg age: {avg_age:.1f})")
                strategy.append("Emphasize quality and premium services")
        
        # Income analysis
        if 'Annual Income (k$)' in features:
            avg_income = cluster_data['Annual Income (k$)'].mean()
            if avg_income < 40:
                income_level = "Low"
                characteristics.append(f"Low income (avg: ${avg_income:.1f}k)")
                strategy.append("Offer budget-friendly options and discounts")
            elif avg_income < 80:
                income_level = "Medium"
                characteristics.append(f"Medium income (avg: ${avg_income:.1f}k)")
                strategy.append("Provide good value for money products")
            else:
                income_level = "High"
                characteristics.append(f"High income (avg: ${avg_income:.1f}k)")
                strategy.append("Target premium and luxury products")
        
        # Spending analysis
        if 'Spending Score (1-100)' in features:
            avg_spending = cluster_data['Spending Score (1-100)'].mean()
            if avg_spending < 35:
                spending_level = "Low"
                characteristics.append(f"Low spending behavior (avg score: {avg_spending:.1f})")
                strategy.append("Incentivize with loyalty programs and special offers")
            elif avg_spending < 65:
                spending_level = "Medium"
                characteristics.append(f"Moderate spending behavior (avg score: {avg_spending:.1f})")
                strategy.append("Maintain engagement with regular promotions")
            else:
                spending_level = "High"
                characteristics.append(f"High spending behavior (avg score: {avg_spending:.1f})")
                strategy.append("Focus on premium experiences and exclusive products")
        
        # Gender analysis
        if 'Genre' in df_clustered.columns:
            gender_dist = cluster_data['Genre'].value_counts()
            dominant_gender = gender_dist.index[0]
            gender_ratio = gender_dist.iloc[0] / cluster_size * 100
            
            if gender_ratio > 70:
                characteristics.append(f"Predominantly {dominant_gender.lower()} ({gender_ratio:.1f}%)")
                if dominant_gender == 'Female':
                    strategy.append("Tailor marketing to female preferences")
                else:
                    strategy.append("Tailor marketing to male preferences")
        
        # Generate profile name
        profile_parts = []
        if 'Age' in features:
            profile_parts.append(age_group)
        if 'Annual Income (k$)' in features:
            profile_parts.append(f"{income_level} Income")
        if 'Spending Score (1-100)' in features:
            profile_parts.append(f"{spending_level} Spenders")
        
        profile = " ".join(profile_parts) if profile_parts else f"Cluster {cluster_id}"
        
        # Generate detailed key characteristics
        detailed_characteristics = []
        
        if 'Age' in features:
            avg_age = cluster_data['Age'].mean()
            age_std = cluster_data['Age'].std()
            detailed_characteristics.append(f"Average age: {avg_age:.1f} years (±{age_std:.1f})")
            
        if 'Annual Income (k$)' in features:
            avg_income = cluster_data['Annual Income (k$)'].mean()
            income_std = cluster_data['Annual Income (k$)'].std()
            detailed_characteristics.append(f"Average income: ${avg_income:.1f}k annually (±${income_std:.1f}k)")
            
        if 'Spending Score (1-100)' in features:
            avg_spending = cluster_data['Spending Score (1-100)'].mean()
            spending_std = cluster_data['Spending Score (1-100)'].std()
            detailed_characteristics.append(f"Spending score: {avg_spending:.1f}/100 (±{spending_std:.1f})")
        
        if 'Genre' in df_clustered.columns:
            gender_dist = cluster_data['Genre'].value_counts()
            gender_breakdown = ", ".join([f"{gender}: {count} ({count/cluster_size*100:.1f}%)" 
                                        for gender, count in gender_dist.items()])
            detailed_characteristics.append(f"Gender distribution: {gender_breakdown}")
        
        # Generate comprehensive marketing strategies
        detailed_strategies = []
        
        # Age-based strategies
        if 'Age' in features:
            avg_age = cluster_data['Age'].mean()
            if avg_age < 30:
                detailed_strategies.extend([
                    "Digital-first marketing: Focus on social media platforms (Instagram, TikTok, Snapchat)",
                    "Mobile optimization: Ensure seamless mobile shopping experience",
                    "Trendy products: Emphasize latest fashion, technology, and lifestyle trends",
                    "Influencer partnerships: Collaborate with young influencers and micro-influencers"
                ])
            elif avg_age < 50:
                detailed_strategies.extend([
                    "Family-focused messaging: Highlight family benefits and value propositions",
                    "Multi-channel approach: Combine digital and traditional marketing channels",
                    "Quality emphasis: Stress product durability and long-term value",
                    "Convenience features: Promote time-saving and efficiency benefits"
                ])
            else:
                detailed_strategies.extend([
                    "Trust-building: Emphasize brand heritage, reliability, and customer service",
                    "Traditional channels: Utilize email, print, and direct mail marketing",
                    "Premium positioning: Focus on quality, craftsmanship, and exclusivity",
                    "Personal service: Offer dedicated support and consultation services"
                ])
        
        # Income-based strategies
        if 'Annual Income (k$)' in features:
            avg_income = cluster_data['Annual Income (k$)'].mean()
            if avg_income < 40:
                detailed_strategies.extend([
                    "Value pricing: Offer competitive prices and budget-friendly options",
                    "Payment flexibility: Provide installment plans and financing options",
                    "Discount programs: Create loyalty rewards and seasonal promotions",
                    "Essential products: Focus on necessary items rather than luxury goods"
                ])
            elif avg_income < 80:
                detailed_strategies.extend([
                    "Balanced approach: Mix of value and premium offerings",
                    "Seasonal campaigns: Target specific occasions and holidays",
                    "Bundle deals: Create attractive package offers and combos",
                    "Quality-price balance: Emphasize good value for money"
                ])
            else:
                detailed_strategies.extend([
                    "Premium positioning: Highlight luxury, exclusivity, and status",
                    "Personalized service: Offer VIP treatment and custom solutions",
                    "High-end products: Focus on premium brands and luxury items",
                    "Exclusive access: Provide early access to new products and special events"
                ])
        
        # Spending-based strategies
        if 'Spending Score (1-100)' in features:
            avg_spending = cluster_data['Spending Score (1-100)'].mean()
            if avg_spending < 35:
                detailed_strategies.extend([
                    "Incentive programs: Implement strong loyalty rewards and cashback offers",
                    "Educational content: Show value and benefits of products",
                    "Limited-time offers: Create urgency with flash sales and time-limited deals",
                    "Free trials: Offer risk-free product testing opportunities"
                ])
            elif avg_spending < 65:
                detailed_strategies.extend([
                    "Regular engagement: Maintain consistent communication and touchpoints",
                    "Seasonal promotions: Align offers with shopping seasons and holidays",
                    "Cross-selling: Recommend complementary products and services",
                    "Feedback loops: Gather insights to improve product offerings"
                ])
            else:
                detailed_strategies.extend([
                    "Premium experiences: Offer exclusive events and VIP shopping experiences",
                    "Early access: Provide first access to new collections and limited editions",
                    "Personalization: Deliver tailored recommendations and custom products",
                    "Concierge services: Offer personal shopping assistance and styling services"
                ])
        
        insights[cluster_id] = {
            'profile': profile,
            'size': cluster_size,
            'percentage': cluster_percentage,
            'characteristics': "; ".join(characteristics),
            'strategy': "; ".join(strategy),
            'detailed_characteristics': detailed_characteristics,
            'detailed_strategies': detailed_strategies,
            'key_metrics': {
                'age': cluster_data['Age'].mean() if 'Age' in features else None,
                'income': cluster_data['Annual Income (k$)'].mean() if 'Annual Income (k$)' in features else None,
                'spending': cluster_data['Spending Score (1-100)'].mean() if 'Spending Score (1-100)' in features else None
            }
        }
    
    return insights

def calculate_optimal_clusters(df, features, max_clusters=10, scale_features=True):
    """
    Calculate the optimal number of clusters using elbow method and silhouette analysis.
    
    Args:
        df (pandas.DataFrame): The dataset
        features (list): Features to use for clustering
        max_clusters (int): Maximum number of clusters to test
        scale_features (bool): Whether to scale features
        
    Returns:
        dict: Dictionary containing WCSS and silhouette scores for different cluster numbers
    """
    # Prepare data
    X = df[features].copy()
    X = X.fillna(X.median())
    
    if scale_features:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values
    
    wcss_values = []
    silhouette_scores = []
    k_range = range(2, max_clusters + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Calculate WCSS
        wcss_values.append(kmeans.inertia_)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    return {
        'k_range': list(k_range),
        'wcss_values': wcss_values,
        'silhouette_scores': silhouette_scores
    }

def get_cluster_statistics(df_clustered, features):
    """
    Get detailed statistics for each cluster.
    
    Args:
        df_clustered (pandas.DataFrame): Dataset with cluster labels
        features (list): Features used for clustering
        
    Returns:
        dict: Dictionary containing statistics for each cluster
    """
    cluster_stats = {}
    n_clusters = df_clustered['Cluster'].nunique()
    
    for cluster_id in range(n_clusters):
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
        
        stats = {
            'size': len(cluster_data),
            'percentage': (len(cluster_data) / len(df_clustered)) * 100
        }
        
        # Calculate statistics for each feature
        for feature in features:
            if feature in cluster_data.columns:
                feature_stats = {
                    'mean': cluster_data[feature].mean(),
                    'median': cluster_data[feature].median(),
                    'std': cluster_data[feature].std(),
                    'min': cluster_data[feature].min(),
                    'max': cluster_data[feature].max(),
                    'q25': cluster_data[feature].quantile(0.25),
                    'q75': cluster_data[feature].quantile(0.75)
                }
                stats[feature] = feature_stats
        
        # Gender distribution if available
        if 'Genre' in cluster_data.columns:
            gender_dist = cluster_data['Genre'].value_counts()
            stats['gender_distribution'] = gender_dist.to_dict()
        
        cluster_stats[cluster_id] = stats
    
    return cluster_stats

def compare_clusters(df_clustered, features):
    """
    Compare clusters across different features.
    
    Args:
        df_clustered (pandas.DataFrame): Dataset with cluster labels
        features (list): Features used for clustering
        
    Returns:
        pandas.DataFrame: Comparison table of clusters
    """
    comparison_data = []
    n_clusters = df_clustered['Cluster'].nunique()
    
    for cluster_id in range(n_clusters):
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
        
        row = {'Cluster': cluster_id, 'Size': len(cluster_data)}
        
        for feature in features:
            if feature in cluster_data.columns:
                row[f'{feature}_mean'] = cluster_data[feature].mean()
                row[f'{feature}_std'] = cluster_data[feature].std()
        
        # Add gender distribution if available
        if 'Genre' in cluster_data.columns:
            gender_dist = cluster_data['Genre'].value_counts()
            row['Male_count'] = gender_dist.get('Male', 0)
            row['Female_count'] = gender_dist.get('Female', 0)
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

def predict_cluster(kmeans_model, scaler, new_data):
    """
    Predict cluster for new data points.
    
    Args:
        kmeans_model: Trained KMeans model
        scaler: Fitted scaler (or None if no scaling was used)
        new_data: New data points to predict
        
    Returns:
        numpy.array: Predicted cluster labels
    """
    if scaler is not None:
        new_data_scaled = scaler.transform(new_data)
    else:
        new_data_scaled = new_data
    
    return kmeans_model.predict(new_data_scaled)
