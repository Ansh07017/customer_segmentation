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
    # Silver color palette
    silver_colors = ['#C0C0C0', '#A8A8A8', '#D3D3D3', '#B8B8B8', '#E8E8E8', '#9E9E9E', '#BEBEBE', '#DCDCDC']
    
    # Ensure Cluster column is string for proper color mapping
    df_viz = df_clustered.copy()
    df_viz['Cluster'] = df_viz['Cluster'].astype(str)
    
    if len(features) == 2:
        # 2D scatter plot
        fig = px.scatter(
            df_viz,
            x=features[0],
            y=features[1],
            color='Cluster',
            title=f'Customer Clusters: {features[0]} vs {features[1]}',
            hover_data=['CustomerID'] if 'CustomerID' in df_viz.columns else None,
            color_discrete_sequence=silver_colors,
            opacity=0.7
        )
        
        # Add cluster centers if available
        if 'Cluster' in df_viz.columns:
            centroids = df_clustered.groupby('Cluster')[features].mean().reset_index()
            fig.add_scatter(
                x=centroids[features[0]],
                y=centroids[features[1]],
                mode='markers',
                marker=dict(size=20, color='#2F2F2F', symbol='x', line=dict(width=3, color='white')),
                name='Centroids',
                showlegend=True
            )
        
    elif len(features) == 3:
        # 3D scatter plot
        fig = px.scatter_3d(
            df_viz,
            x=features[0],
            y=features[1],
            z=features[2],
            color='Cluster',
            title=f'Customer Clusters: {features[0]} vs {features[1]} vs {features[2]}',
            hover_data=['CustomerID'] if 'CustomerID' in df_viz.columns else None,
            color_discrete_sequence=silver_colors,
            opacity=0.7
        )
        
        # Add cluster centers
        if 'Cluster' in df_viz.columns:
            centroids = df_clustered.groupby('Cluster')[features].mean().reset_index()
            fig.add_scatter3d(
                x=centroids[features[0]],
                y=centroids[features[1]],
                z=centroids[features[2]],
                mode='markers',
                marker=dict(size=15, color='#2F2F2F', symbol='x'),
                name='Centroids',
                showlegend=True
            )
    
    else:
        # For more than 3 features, create a parallel coordinates plot
        feature_cols = features + ['Cluster']
        df_parallel = df_viz[feature_cols].copy()
        fig = px.parallel_coordinates(
            df_parallel,
            color='Cluster',
            title='Customer Clusters - Parallel Coordinates',
            color_continuous_scale=silver_colors[:5]
        )
    
    # Apply dark theme with silver accents
    fig.update_layout(
        height=600,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12),
        title=dict(font=dict(color='white', size=16)),
        xaxis=dict(gridcolor='rgba(192,192,192,0.2)', color='white'),
        yaxis=dict(gridcolor='rgba(192,192,192,0.2)', color='white'),
        legend=dict(font=dict(color='white'))
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
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) < 2:
        # Create a simple message figure if not enough numerical columns
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough numerical features for correlation analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Calculate correlation matrix
    correlation_matrix = df[numerical_cols].corr()
    
    # Create heatmap
    fig = px.imshow(
        correlation_matrix,
        labels=dict(x="Features", y="Features", color="Correlation"),
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        color_continuous_scale='RdBu',
        aspect="auto",
        title="Feature Correlation Matrix"
    )
    
    # Add correlation values as text
    fig.update_traces(
        text=np.around(correlation_matrix.values, decimals=2),
        texttemplate="%{text}",
        textfont={"size": 10}
    )
    
    fig.update_layout(
        height=500,
        width=500
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
