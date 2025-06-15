import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

def create_cluster_visualization(df_clustered, features):
    """
    Create cluster visualization based on the selected features.
    
    Args:
        df_clustered (pandas.DataFrame): Dataset with cluster labels
        features (list): Features used for clustering
        
    Returns:
        plotly.graph_objects.Figure: Cluster visualization
    """
    # Golden and purple theme palette
    theme_colors = ['#9b59d0', '#fbbf24', '#b983e0', '#fcd34d', '#7c3aed', '#f59e0b', '#c084fc', '#fde047']
    
    # Ensure we have the data and cluster column
    if df_clustered is None or df_clustered.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, 
                          font=dict(size=16, color='white'))
        return fig
    
    # Ensure Cluster column exists and is properly formatted
    if 'Cluster' not in df_clustered.columns:
        fig = go.Figure()
        fig.add_annotation(text="No cluster data available", xref="paper", yref="paper", x=0.5, y=0.5,
                          font=dict(size=16, color='white'))
        return fig
    
    # Create a clean copy of the data and ensure proper format
    df_viz = df_clustered.copy().reset_index(drop=True)
    df_viz['Cluster'] = df_viz['Cluster'].astype(str)  # Keep as string for color mapping
    
    print(f"Creating visualization with {len(df_viz)} data points, {df_viz['Cluster'].nunique()} clusters")
    
    # Use plotly express for reliable data handling - this ensures all data points are included
    fig = px.scatter(
        df_viz,
        x=features[0],
        y=features[1],
        color='Cluster',
        title=f'Customer Clusters: {features[0]} vs {features[1]}',
        color_discrete_sequence=theme_colors,
        opacity=0.8
    )
    
    # Enhance markers for better visibility
    fig.update_traces(marker=dict(size=10, line=dict(width=2, color='white')))
    
    # Add cluster centers if we have 2D data
    if len(features) >= 2:
        centroids = df_clustered.groupby('Cluster')[features].mean().reset_index()
        
        for idx, row in centroids.iterrows():
            fig.add_trace(go.Scatter(
                x=[row[features[0]]],
                y=[row[features[1]]],
                mode='markers',
                name=f'Centroid {int(row["Cluster"])}',
                marker=dict(
                    size=25,
                    color='black',
                    symbol='x',
                    line=dict(width=4, color='white')
                ),
                showlegend=True
            ))
        

    
    # Apply dark theme with enhanced visibility
    fig.update_layout(
        height=500,
        showlegend=True,
        plot_bgcolor='rgba(26, 22, 37, 0.8)',
        paper_bgcolor='rgba(26, 22, 37, 0.8)',
        font=dict(color='#ffffff', size=13, family='Arial Black'),
        title=dict(font=dict(color='#ffffff', size=18, family='Arial Black')),
        xaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.3)', 
            color='#ffffff',
            title_font=dict(color='#ffffff', size=14, family='Arial Black'),
            tickfont=dict(size=12, color='#ffffff', family='Arial Black'),
            linewidth=2,
            linecolor='#ffffff'
        ),
        yaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.3)', 
            color='#ffffff',
            title_font=dict(color='#ffffff', size=14, family='Arial Black'),
            tickfont=dict(size=12, color='#ffffff', family='Arial Black'),
            linewidth=2,
            linecolor='#ffffff'
        ),
        legend=dict(
            font=dict(color='#ffffff', size=12, family='Arial Black'), 
            bgcolor='rgba(42, 31, 61, 0.9)',
            bordercolor='rgba(255, 255, 255, 0.5)',
            borderwidth=2
        ),
        margin=dict(l=70, r=70, t=80, b=60)
    )
    
    return fig

def create_correlation_heatmap(df):
    """
    Create a correlation heatmap for numerical features.
    
    Args:
        df (pandas.DataFrame): The dataset
        
    Returns:
        plotly.graph_objects.Figure: Correlation heatmap
    """
    # Select specific numerical columns for correlation analysis
    numerical_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    
    # Filter to only columns that exist in the dataframe
    available_cols = [col for col in numerical_cols if col in df.columns]
    
    if len(available_cols) < 2:
        # Create a simple message figure if not enough numerical columns
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough numerical features for correlation analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color='white')
        )
        fig.update_layout(
            plot_bgcolor='rgba(26, 22, 37, 0.8)',
            paper_bgcolor='rgba(26, 22, 37, 0.8)'
        )
        return fig
    
    # Calculate correlation matrix
    correlation_matrix = df[available_cols].corr()
    
    # Create heatmap using go.Heatmap for better control
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale=[[0, '#9b59d0'], [0.5, '#2a1e25'], [1, '#fbbf24']],
        text=np.around(correlation_matrix.values, decimals=2),
        texttemplate="%{text}",
        textfont={"size": 12, "color": "white", "family": "Arial Black"},
        hoverongaps=False,
        colorbar=dict(
            title=dict(text="Correlation", font=dict(color='#ffffff', size=14, family='Arial Black')),
            tickfont=dict(color='#ffffff', size=12, family='Arial Black')
        )
    ))
    
    fig.update_layout(
        title=dict(text="Feature Correlation Matrix", font=dict(color='#ffffff', size=18, family='Arial Black')),
        height=480,
        width=550,
        plot_bgcolor='rgba(26, 22, 37, 0.8)',
        paper_bgcolor='rgba(26, 22, 37, 0.8)',
        font=dict(color='#ffffff', size=13, family='Arial Black'),
        xaxis=dict(
            color='#ffffff',
            tickfont=dict(size=12, color='#ffffff', family='Arial Black'),
            side='bottom'
        ),
        yaxis=dict(
            color='#ffffff',
            tickfont=dict(size=12, color='#ffffff', family='Arial Black'),
            autorange='reversed'
        ),
        margin=dict(l=100, r=100, t=80, b=60)
    )
    
    return fig

def create_cluster_comparison_charts(df_clustered, features):
    """
    Create comparison charts for clusters across different features.
    
    Args:
        df_clustered (pandas.DataFrame): Dataset with cluster labels
        features (list): Features to compare
        
    Returns:
        dict: Dictionary of comparison charts
    """
    charts = {}
    
    for feature in features:
        if feature in df_clustered.columns:
            # Box plot comparing feature across clusters
            fig = px.box(
                df_clustered,
                x='Cluster',
                y=feature,
                title=f'{feature} Distribution by Cluster',
                color='Cluster',
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            
            fig.update_layout(
                showlegend=False,
                height=400,
                xaxis_title='Cluster',
                yaxis_title=feature
            )
            
            charts[f'{feature}_boxplot'] = fig
            
            # Violin plot for more detailed distribution
            fig_violin = px.violin(
                df_clustered,
                x='Cluster',
                y=feature,
                title=f'{feature} Distribution by Cluster (Detailed)',
                color='Cluster',
                color_discrete_sequence=px.colors.qualitative.Set1,
                box=True
            )
            
            fig_violin.update_layout(
                showlegend=False,
                height=400,
                xaxis_title='Cluster',
                yaxis_title=feature
            )
            
            charts[f'{feature}_violin'] = fig_violin
    
    return charts

def create_cluster_profile_radar(df_clustered, features):
    """
    Create radar charts for cluster profiles.
    
    Args:
        df_clustered (pandas.DataFrame): Dataset with cluster labels
        features (list): Features to include in radar chart
        
    Returns:
        plotly.graph_objects.Figure: Radar chart
    """
    # Calculate mean values for each cluster
    cluster_means = df_clustered.groupby('Cluster')[features].mean()
    
    # Normalize values to 0-1 scale for better visualization
    normalized_means = cluster_means.copy()
    for feature in features:
        min_val = df_clustered[feature].min()
        max_val = df_clustered[feature].max()
        normalized_means[feature] = (cluster_means[feature] - min_val) / (max_val - min_val)
    
    # Create radar chart
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, cluster_id in enumerate(normalized_means.index):
        values = normalized_means.loc[cluster_id].values.tolist()
        values += [values[0]]  # Close the radar chart
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=features + [features[0]],
            fill='toself',
            name=f'Cluster {cluster_id}',
            line_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Cluster Profiles - Radar Chart",
        height=600
    )
    
    return fig

def create_cluster_size_chart(df_clustered):
    """
    Create a chart showing cluster sizes.
    
    Args:
        df_clustered (pandas.DataFrame): Dataset with cluster labels
        
    Returns:
        plotly.graph_objects.Figure: Cluster size chart
    """
    cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
    
    fig = go.Figure()
    
    # Bar chart
    fig.add_trace(go.Bar(
        x=[f'Cluster {i}' for i in cluster_counts.index],
        y=cluster_counts.values,
        marker_color=px.colors.qualitative.Set1[:len(cluster_counts)],
        text=cluster_counts.values,
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Customer Distribution Across Clusters',
        xaxis_title='Cluster',
        yaxis_title='Number of Customers',
        height=400,
        showlegend=False
    )
    
    return fig

def create_feature_importance_chart(df_clustered, features):
    """
    Create a chart showing feature importance in clustering.
    
    Args:
        df_clustered (pandas.DataFrame): Dataset with cluster labels
        features (list): Features used in clustering
        
    Returns:
        plotly.graph_objects.Figure: Feature importance chart
    """
    # Calculate variance within clusters for each feature
    feature_importance = {}
    
    for feature in features:
        if feature in df_clustered.columns:
            # Calculate within-cluster variance
            total_variance = 0
            total_samples = 0
            
            for cluster_id in df_clustered['Cluster'].unique():
                cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id][feature]
                cluster_variance = cluster_data.var()
                cluster_size = len(cluster_data)
                
                total_variance += cluster_variance * cluster_size
                total_samples += cluster_size
            
            within_cluster_variance = total_variance / total_samples
            overall_variance = df_clustered[feature].var()
            
            # Higher importance means lower within-cluster variance relative to overall variance
            importance = 1 - (within_cluster_variance / overall_variance)
            feature_importance[feature] = importance
    
    # Create bar chart
    fig = px.bar(
        x=list(feature_importance.keys()),
        y=list(feature_importance.values()),
        title='Feature Importance in Clustering',
        labels={'x': 'Features', 'y': 'Importance Score'},
        color=list(feature_importance.values()),
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False
    )
    
    return fig

def create_cluster_transition_matrix(df_clustered_old, df_clustered_new):
    """
    Create a transition matrix showing how customers move between clusters.
    
    Args:
        df_clustered_old (pandas.DataFrame): Previous clustering results
        df_clustered_new (pandas.DataFrame): New clustering results
        
    Returns:
        plotly.graph_objects.Figure: Transition matrix heatmap
    """
    # Create transition matrix
    transition_matrix = pd.crosstab(
        df_clustered_old['Cluster'],
        df_clustered_new['Cluster'],
        normalize='index'
    )
    
    # Create heatmap
    fig = px.imshow(
        transition_matrix,
        labels=dict(x="New Cluster", y="Old Cluster", color="Transition Probability"),
        title="Cluster Transition Matrix",
        color_continuous_scale='Blues',
        aspect="auto"
    )
    
    # Add percentage values as text
    fig.update_traces(
        text=np.around(transition_matrix.values * 100, decimals=1),
        texttemplate="%{text}%",
        textfont={"size": 10}
    )
    
    fig.update_layout(height=400)
    
    return fig

def create_silhouette_plot(df_clustered, features, cluster_labels):
    """
    Create a silhouette plot for cluster analysis.
    
    Args:
        df_clustered (pandas.DataFrame): Dataset with cluster labels
        features (list): Features used for clustering
        cluster_labels (array): Cluster labels
        
    Returns:
        plotly.graph_objects.Figure: Silhouette plot
    """
    from sklearn.metrics import silhouette_samples
    from sklearn.preprocessing import StandardScaler
    
    # Prepare data
    X = df_clustered[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calculate silhouette scores for each sample
    silhouette_scores = silhouette_samples(X_scaled, cluster_labels)
    
    # Create silhouette plot
    fig = go.Figure()
    
    y_lower = 10
    colors = px.colors.qualitative.Set1
    
    for i in range(len(np.unique(cluster_labels))):
        cluster_silhouette_scores = silhouette_scores[cluster_labels == i]
        cluster_silhouette_scores.sort()
        
        size_cluster_i = cluster_silhouette_scores.shape[0]
        y_upper = y_lower + size_cluster_i
        
        fig.add_trace(go.Scatter(
            x=cluster_silhouette_scores,
            y=np.arange(y_lower, y_upper),
            mode='lines',
            fill='tonexty' if i > 0 else 'tozeroy',
            name=f'Cluster {i}',
            line=dict(color=colors[i % len(colors)], width=0),
            fillcolor=colors[i % len(colors)]
        ))
        
        y_lower = y_upper + 10
    
    # Add average silhouette score line
    avg_silhouette_score = silhouette_scores.mean()
    fig.add_vline(
        x=avg_silhouette_score,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Average Score: {avg_silhouette_score:.3f}"
    )
    
    fig.update_layout(
        title='Silhouette Plot for Clusters',
        xaxis_title='Silhouette Coefficient Values',
        yaxis_title='Cluster Label',
        height=600,
        showlegend=True
    )
    
    return fig
