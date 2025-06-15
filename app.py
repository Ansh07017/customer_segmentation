from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Import utility functions
from utils.data_loader import get_sample_data, load_data
from utils.eda_functions import perform_eda, get_statistical_summary, detect_outliers
from utils.clustering import perform_clustering, get_cluster_insights, calculate_optimal_clusters
from utils.visualizations import create_cluster_visualization, create_correlation_heatmap

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables to store data and models
current_data = None
clustering_results = None
trained_model = None
model_scaler = None
feature_columns = None

def initialize_app():
    """Initialize the application with pre-trained model"""
    global current_data, clustering_results, trained_model, model_scaler, feature_columns
    
    try:
        # Load the Mall Customers dataset
        current_data = get_sample_data()
        print(f"Loaded {len(current_data)} customer records from Mall_Customers.csv")
        
        # Perform clustering with optimal parameters
        clustering_results = perform_clustering(
            current_data, 
            ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'], 
            5,  # 5 clusters works well for this dataset
            True,  # scale features
            42  # random state
        )
        print("Clustering analysis completed")
        
        # Train the classification model
        df_clustered = clustering_results['df_clustered']
        features_used = clustering_results['features_used']
        
        # Prepare features for training
        feature_data = df_clustered[features_used].copy()
        
        # Add gender encoding
        if 'Genre' in df_clustered.columns:
            feature_data['Genre_Male'] = (df_clustered['Genre'] == 'Male').astype(int)
            feature_data['Genre_Female'] = (df_clustered['Genre'] == 'Female').astype(int)
        
        feature_columns = feature_data.columns.tolist()
        
        # Scale features
        model_scaler = StandardScaler()
        X_scaled = model_scaler.fit_transform(feature_data)
        
        # Get cluster labels as target
        y = df_clustered['Cluster'].values
        
        # Train Random Forest classifier
        trained_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        trained_model.fit(X_scaled, y)
        
        # Calculate accuracy on full dataset for verification
        y_pred = trained_model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        
        # Save model to disk
        os.makedirs('models', exist_ok=True)
        with open('models/customer_classifier.pkl', 'wb') as f:
            pickle.dump({
                'model': trained_model,
                'scaler': model_scaler,
                'feature_columns': feature_columns,
                'features_used': features_used,
                'clustering_results': clustering_results
            }, f)
        
        print(f"Classification model trained and saved with accuracy: {accuracy:.2%}")
        print("Application ready for customer classification!")
        
    except Exception as e:
        print(f"Error during initialization: {e}")
        # Try to load existing model if available
        try:
            if os.path.exists('models/customer_classifier.pkl'):
                with open('models/customer_classifier.pkl', 'rb') as f:
                    saved_data = pickle.load(f)
                    trained_model = saved_data['model']
                    model_scaler = saved_data['scaler']
                    feature_columns = saved_data['feature_columns']
                    if 'clustering_results' in saved_data:
                        clustering_results = saved_data['clustering_results']
                print("Loaded existing model from disk")
        except Exception as load_error:
            print(f"Error loading existing model: {load_error}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload_data', methods=['POST'])
def upload_data():
    global current_data
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if file.filename is None or not file.filename.lower().endswith('.csv'):
            return jsonify({'success': False, 'error': 'Only CSV files are supported'})
        
        # Read the uploaded CSV file
        import io
        file_content = file.read()
        file.seek(0)  # Reset file pointer
        current_data = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
        
        # Try to standardize column names for common customer segmentation datasets
        column_mapping = {}
        for col in current_data.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['customer', 'id']):
                column_mapping[col] = 'CustomerID'
            elif any(keyword in col_lower for keyword in ['gender', 'genre', 'sex']):
                column_mapping[col] = 'Genre'
            elif 'age' in col_lower:
                column_mapping[col] = 'Age'
            elif any(keyword in col_lower for keyword in ['income', 'annual']):
                column_mapping[col] = 'Annual Income (k$)'
            elif any(keyword in col_lower for keyword in ['spending', 'score']):
                column_mapping[col] = 'Spending Score (1-100)'
        
        # Apply column mapping
        current_data = current_data.rename(columns=column_mapping)
        
        # Clean the data
        current_data = current_data.dropna()
        
        # Standardize Genre values
        if 'Genre' in current_data.columns:
            current_data['Genre'] = current_data['Genre'].str.title()
            current_data['Genre'] = current_data['Genre'].replace({'M': 'Male', 'F': 'Female'})
        
        return jsonify({
            'success': True,
            'message': 'Data uploaded successfully',
            'data_info': {
                'rows': len(current_data),
                'columns': len(current_data.columns),
                'column_names': current_data.columns.tolist()
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/load_sample_data')
def load_sample_data():
    global current_data
    try:
        current_data = get_sample_data()
        return jsonify({
            'success': True,
            'message': 'Mall Customers dataset loaded successfully',
            'data_info': {
                'rows': len(current_data),
                'columns': len(current_data.columns),
                'column_names': current_data.columns.tolist(),
                'source': 'Mall_Customers.csv from uploaded dataset'
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_data_overview')
def get_data_overview():
    global current_data
    if current_data is None:
        return jsonify({'success': False, 'error': 'No data loaded'})
    
    try:
        # Basic statistics
        overview = {
            'total_customers': len(current_data),
            'features': len(current_data.columns),
            'male_customers': len(current_data[current_data['Genre'] == 'Male']),
            'female_customers': len(current_data[current_data['Genre'] == 'Female']),
            'summary_stats': current_data.describe().to_dict(),
            'data_preview': current_data.head(10).to_dict('records'),
            'data_types': current_data.dtypes.astype(str).to_dict(),
            'missing_values': current_data.isnull().sum().to_dict()
        }
        return jsonify({'success': True, 'data': overview})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_eda')
def get_eda():
    global current_data
    if current_data is None:
        return jsonify({'success': False, 'error': 'No data loaded'})
    
    try:
        # Perform EDA
        eda_results = perform_eda(current_data)
        
        # Convert plotly figures to JSON
        eda_json = {}
        for key, fig in eda_results.items():
            eda_json[key] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Correlation heatmap
        correlation_fig = create_correlation_heatmap(current_data)
        eda_json['correlation'] = json.dumps(correlation_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Statistical summary
        stats_summary = get_statistical_summary(current_data)
        
        # Outlier detection
        outliers = detect_outliers(current_data)
        
        # Convert numpy/pandas types to Python native types for JSON serialization
        def convert_to_serializable(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        return jsonify({
            'success': True,
            'plots': eda_json,
            'statistics': {
                'numerical': convert_to_serializable(stats_summary['numerical'].to_dict()),
                'categorical': convert_to_serializable(stats_summary['categorical'])
            },
            'outliers': convert_to_serializable(outliers)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/perform_clustering', methods=['POST'])
def api_perform_clustering():
    global current_data, clustering_results
    if current_data is None:
        return jsonify({'success': False, 'error': 'No data loaded'})
    
    try:
        data = request.get_json()
        n_clusters = data.get('n_clusters', 5)
        features = data.get('features', ['Annual Income (k$)', 'Spending Score (1-100)'])
        scale_features = data.get('scale_features', True)
        random_state = data.get('random_state', 42)
        
        # Validate features
        missing_features = [f for f in features if f not in current_data.columns]
        if missing_features:
            return jsonify({
                'success': False, 
                'error': f'Missing features: {missing_features}'
            })
        
        # Perform clustering
        clustering_results = perform_clustering(
            current_data, features, n_clusters, scale_features, random_state
        )
        
        # Get cluster insights
        insights = get_cluster_insights(clustering_results['df_clustered'], features)
        
        # Create visualization
        cluster_viz = create_cluster_visualization(clustering_results['df_clustered'], features)
        
        # Cluster distribution
        cluster_counts = clustering_results['df_clustered']['Cluster'].value_counts().sort_index()
        
        # Create meaningful cluster labels based on insights
        cluster_labels = []
        for i in cluster_counts.index:
            if str(i) in insights:
                profile = insights[str(i)].get('profile', f'Cluster {i}')
                cluster_labels.append(f"{profile} ({cluster_counts[i]} customers)")
            else:
                cluster_labels.append(f"Cluster {i} ({cluster_counts[i]} customers)")
        
        # Golden and purple theme palette
        theme_colors = ['#9b59d0', '#fbbf24', '#b983e0', '#fcd34d', '#7c3aed', '#f59e0b', '#c084fc', '#fde047']
        
        # Enhanced Pie chart with meaningful labels
        pie_fig = px.pie(
            values=cluster_counts.values,
            names=cluster_labels,
            title="Customer Segment Distribution",
            color_discrete_sequence=theme_colors
        )
        
        pie_fig.update_layout(
            height=550,
            plot_bgcolor='rgba(26, 22, 37, 0.8)',
            paper_bgcolor='rgba(26, 22, 37, 0.8)',
            font=dict(color='#f3f4f6', size=12),
            title=dict(font=dict(color='#ffffff', size=16, family='Arial Black')),
            legend=dict(
                font=dict(color='#ffffff', size=11), 
                bgcolor='rgba(42, 31, 61, 0.9)',
                bordercolor='rgba(111, 66, 193, 0.3)',
                borderwidth=1
            ),
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        # Enhanced Bar chart with percentages
        percentages = [(count / len(clustering_results['df_clustered'])) * 100 for count in cluster_counts.values]
        bar_labels = [f"Segment {i+1}" for i in range(len(cluster_counts))]
        
        bar_fig = px.bar(
            x=bar_labels,
            y=list(cluster_counts.values),
            title="Customer Segment Sizes",
            text=[f"{count}<br>({pct:.1f}%)" for count, pct in zip(cluster_counts.values, percentages)]
        )
        
        # Update bar colors manually to ensure visibility
        bar_fig.update_traces(
            marker_color=theme_colors[:len(bar_labels)],
            marker_line_color='white',
            marker_line_width=2,
            textposition='outside',
            textfont=dict(size=14, color='white', family='Arial Black')
        )
        
        bar_fig.update_layout(
            height=450,
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
                title='Number of Customers',
                title_font=dict(color='#ffffff', size=14, family='Arial Black'),
                tickfont=dict(size=12, color='#ffffff', family='Arial Black'),
                linewidth=2,
                linecolor='#ffffff'
            ),
            showlegend=False,
            margin=dict(l=70, r=50, t=80, b=60)
        )
        
        bar_fig.update_traces(
            textposition='inside', 
            textfont=dict(color='white', size=14, family='Arial Black'),
            marker=dict(line=dict(color='white', width=2))
        )
        
        # Create cluster comparison chart showing feature averages
        cluster_stats = clustering_results['df_clustered'].groupby('Cluster')[features].mean().reset_index()
        
        # Create radar chart for cluster profiles
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        radar_fig = go.Figure()
        
        for idx, cluster_id in enumerate(cluster_stats['Cluster']):
            values = []
            for feature in features:
                # Normalize values to 0-100 scale for better visualization
                feature_values = clustering_results['df_clustered'][feature]
                normalized_val = ((cluster_stats.loc[cluster_stats['Cluster'] == cluster_id, feature].iloc[0] - feature_values.min()) / 
                                (feature_values.max() - feature_values.min())) * 100
                values.append(normalized_val)
            
            # Close the radar chart
            values.append(values[0])
            feature_labels = features + [features[0]]
            
            radar_fig.add_trace(go.Scatterpolar(
                r=values,
                theta=feature_labels,
                fill='toself',
                name=f'Segment {cluster_id + 1}',
                line=dict(color=theme_colors[idx % len(theme_colors)], width=4),
                fillcolor=theme_colors[idx % len(theme_colors)],
                opacity=0.7,
                marker=dict(size=8, color=theme_colors[idx % len(theme_colors)])
            ))
        
        radar_fig.update_layout(
            height=500,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    color='#ffffff',
                    gridcolor='rgba(255, 255, 255, 0.4)',
                    linewidth=2,
                    tickfont=dict(size=12, color='#ffffff')
                ),
                angularaxis=dict(
                    color='#ffffff',
                    linewidth=2,
                    tickfont=dict(size=13, color='#ffffff', family='Arial Black')
                ),
                bgcolor='rgba(26, 22, 37, 0.3)'
            ),
            showlegend=True,
            plot_bgcolor='rgba(26, 22, 37, 0.8)',
            paper_bgcolor='rgba(26, 22, 37, 0.8)',
            font=dict(color='#ffffff', size=13, family='Arial Black'),
            title=dict(text="Customer Segment Profiles", font=dict(color='#ffffff', size=18, family='Arial Black')),
            legend=dict(
                font=dict(color='#ffffff', size=12, family='Arial Black'), 
                bgcolor='rgba(42, 31, 61, 0.9)',
                bordercolor='rgba(255, 255, 255, 0.5)',
                borderwidth=2
            ),
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        # Create feature importance chart with golden-purple gradient
        std_values = [clustering_results['df_clustered'][feature].std() for feature in features]
        feature_importance_fig = px.bar(
            x=features,
            y=std_values,
            title="Feature Variation Across Clusters",
            text=[f"{val:.1f}" for val in std_values]
        )
        
        # Update bar colors and styling manually
        feature_importance_fig.update_traces(
            marker_color=['#9b59d0', '#fbbf24', '#b983e0', '#fcd34d'][:len(features)],
            marker_line_color='white',
            marker_line_width=2,
            textposition='outside',
            textfont=dict(size=14, color='white', family='Arial Black')
        )
        
        feature_importance_fig.update_layout(
            height=450,
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
                title='Standard Deviation',
                title_font=dict(color='#ffffff', size=14, family='Arial Black'),
                tickfont=dict(size=12, color='#ffffff', family='Arial Black'),
                linewidth=2,
                linecolor='#ffffff'
            ),
            showlegend=False,
            margin=dict(l=70, r=50, t=80, b=60)
        )
        
        feature_importance_fig.update_traces(
            textposition='outside', 
            textfont=dict(color='white', size=14, family='Arial Black'),
            marker=dict(line=dict(color='white', width=2))
        )
        
        # Try JSON serialization with error handling
        try:
            json_results = {
                'inertia': clustering_results['inertia'],
                'silhouette_score': clustering_results['silhouette_score'],
                'cluster_visualization': json.dumps(cluster_viz, cls=plotly.utils.PlotlyJSONEncoder),
                'cluster_pie': json.dumps(pie_fig, cls=plotly.utils.PlotlyJSONEncoder),
                'cluster_bar': json.dumps(bar_fig, cls=plotly.utils.PlotlyJSONEncoder),
                'cluster_radar': json.dumps(radar_fig, cls=plotly.utils.PlotlyJSONEncoder),
                'feature_importance': json.dumps(feature_importance_fig, cls=plotly.utils.PlotlyJSONEncoder),
                'insights': insights,
                'cluster_counts': cluster_counts.to_dict(),
                'features_used': features,
                'n_clusters': n_clusters
            }
        except Exception as json_error:
            import traceback
            print(f"JSON serialization error: {json_error}")
            traceback.print_exc()
            return jsonify({'success': False, 'error': f'JSON error: {str(json_error)}'})
        
        return jsonify({
            'success': True,
            'results': json_results
        })
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"CLUSTERING ERROR: {e}")
        print(f"FULL TRACEBACK:\n{error_traceback}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_cluster_analysis')
def get_cluster_analysis():
    global clustering_results
    if clustering_results is None:
        return jsonify({'success': False, 'error': 'No clustering results available'})
    
    try:
        df_clustered = clustering_results['df_clustered']
        features_used = clustering_results['features_used']
        n_clusters = df_clustered['Cluster'].nunique()
        
        # Get detailed cluster statistics
        cluster_details = {}
        for cluster_id in range(n_clusters):
            cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
            
            cluster_stats = {
                'size': len(cluster_data),
                'percentage': (len(cluster_data) / len(df_clustered)) * 100
            }
            
            # Feature statistics
            for feature in features_used:
                if feature in cluster_data.columns:
                    cluster_stats[f'{feature}_mean'] = float(cluster_data[feature].mean())
                    cluster_stats[f'{feature}_std'] = float(cluster_data[feature].std())
                    cluster_stats[f'{feature}_min'] = float(cluster_data[feature].min())
                    cluster_stats[f'{feature}_max'] = float(cluster_data[feature].max())
            
            # Gender distribution
            if 'Genre' in cluster_data.columns:
                gender_dist = cluster_data['Genre'].value_counts()
                cluster_stats['male_count'] = int(gender_dist.get('Male', 0))
                cluster_stats['female_count'] = int(gender_dist.get('Female', 0))
                
                # Gender pie chart for this cluster
                gender_pie = px.pie(
                    values=gender_dist.values,
                    names=gender_dist.index,
                    title=f"Gender Distribution - Cluster {cluster_id}"
                )
                cluster_stats['gender_chart'] = json.dumps(gender_pie, cls=plotly.utils.PlotlyJSONEncoder)
            
            cluster_details[cluster_id] = cluster_stats
        
        # Get cluster insights
        insights = get_cluster_insights(df_clustered, features_used)
        
        return jsonify({
            'success': True,
            'cluster_details': cluster_details,
            'insights': insights,
            'features_used': features_used
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_optimal_clusters', methods=['POST'])
def get_optimal_clusters():
    global current_data
    if current_data is None:
        return jsonify({'success': False, 'error': 'No data loaded'})
    
    try:
        data = request.get_json()
        features = data.get('features', ['Annual Income (k$)', 'Spending Score (1-100)'])
        max_clusters = data.get('max_clusters', 10)
        scale_features = data.get('scale_features', True)
        
        # Calculate optimal clusters
        optimal_results = calculate_optimal_clusters(
            current_data, features, max_clusters, scale_features
        )
        
        # Create elbow plot
        elbow_fig = go.Figure()
        elbow_fig.add_trace(go.Scatter(
            x=optimal_results['k_range'],
            y=optimal_results['wcss_values'],
            mode='lines+markers',
            name='WCSS',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        elbow_fig.update_layout(
            title='Elbow Method for Optimal Number of Clusters',
            xaxis_title='Number of Clusters (k)',
            yaxis_title='Within-Cluster Sum of Squares (WCSS)',
            showlegend=False,
            height=500
        )
        
        # Create silhouette plot
        silhouette_fig = go.Figure()
        silhouette_fig.add_trace(go.Scatter(
            x=optimal_results['k_range'],
            y=optimal_results['silhouette_scores'],
            mode='lines+markers',
            name='Silhouette Score',
            line=dict(color='green', width=2),
            marker=dict(size=8)
        ))
        
        silhouette_fig.update_layout(
            title='Silhouette Analysis for Optimal Number of Clusters',
            xaxis_title='Number of Clusters (k)',
            yaxis_title='Average Silhouette Score',
            showlegend=False,
            height=500
        )
        
        return jsonify({
            'success': True,
            'elbow_plot': json.dumps(elbow_fig, cls=plotly.utils.PlotlyJSONEncoder),
            'silhouette_plot': json.dumps(silhouette_fig, cls=plotly.utils.PlotlyJSONEncoder),
            'optimal_results': optimal_results
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/train_model', methods=['POST'])
def train_model():
    global clustering_results, trained_model, model_scaler, feature_columns
    if clustering_results is None:
        return jsonify({'success': False, 'error': 'No clustering results available'})
    
    try:
        df_clustered = clustering_results['df_clustered']
        features_used = clustering_results['features_used']
        
        # Prepare features for training
        feature_data = df_clustered[features_used].copy()
        
        # Add gender encoding if Genre is available
        if 'Genre' in df_clustered.columns:
            feature_data['Genre_Male'] = (df_clustered['Genre'] == 'Male').astype(int)
            feature_data['Genre_Female'] = (df_clustered['Genre'] == 'Female').astype(int)
        
        # Store feature columns for later use
        feature_columns = feature_data.columns.tolist()
        
        # Scale features
        model_scaler = StandardScaler()
        X_scaled = model_scaler.fit_transform(feature_data)
        
        # Get cluster labels as target
        y = df_clustered['Cluster'].values
        
        # Split data for training and validation
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest classifier
        trained_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        trained_model.fit(X_train, y_train)
        
        # Calculate accuracy
        y_pred = trained_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save model to disk
        os.makedirs('models', exist_ok=True)
        with open('models/customer_classifier.pkl', 'wb') as f:
            pickle.dump({
                'model': trained_model,
                'scaler': model_scaler,
                'feature_columns': feature_columns,
                'features_used': features_used
            }, f)
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'accuracy': accuracy,
            'feature_columns': feature_columns,
            'n_features': len(feature_columns)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/classify_customer', methods=['POST'])
def classify_customer():
    global trained_model, model_scaler, feature_columns, clustering_results
    if trained_model is None or model_scaler is None or feature_columns is None or clustering_results is None:
        return jsonify({'success': False, 'error': 'No trained model available'})
    
    try:
        customer_data = request.get_json()
        if not customer_data:
            return jsonify({'success': False, 'error': 'No customer data provided'})
        
        # Prepare customer data for prediction
        customer_features = {}
        
        # Add numerical features
        for feature in clustering_results['features_used']:
            if feature in customer_data:
                customer_features[feature] = customer_data[feature]
        
        # Add gender encoding if available
        if 'Genre' in customer_data:
            customer_features['Genre_Male'] = 1 if customer_data['Genre'] == 'Male' else 0
            customer_features['Genre_Female'] = 1 if customer_data['Genre'] == 'Female' else 0
        
        # Create feature vector in the same order as training
        feature_vector = []
        for col in feature_columns:
            if col in customer_features:
                feature_vector.append(customer_features[col])
            else:
                feature_vector.append(0)  # Default value for missing features
        
        # Scale features
        feature_vector_array = np.array(feature_vector).reshape(1, -1)
        feature_vector_scaled = model_scaler.transform(feature_vector_array)
        
        # Make prediction
        predicted_cluster = trained_model.predict(feature_vector_scaled)[0]
        prediction_proba = trained_model.predict_proba(feature_vector_scaled)[0]
        confidence = prediction_proba.max()
        
        # Get cluster insights
        insights = get_cluster_insights(clustering_results['df_clustered'], clustering_results['features_used'])
        cluster_insight = insights.get(predicted_cluster, {})
        
        return jsonify({
            'success': True,
            'prediction': {
                'cluster': int(predicted_cluster),
                'confidence': float(confidence),
                'profile': cluster_insight.get('profile', f'Cluster {predicted_cluster}'),
                'characteristics': cluster_insight.get('characteristics', 'No characteristics available'),
                'strategy': cluster_insight.get('strategy', 'No strategy available'),
                'probabilities': {f'Cluster {i}': float(prob) for i, prob in enumerate(prediction_proba)}
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/model_status')
def model_status():
    return jsonify({
        'data_loaded': current_data is not None,
        'clustering_done': clustering_results is not None,
        'model_trained': trained_model is not None,
        'feature_columns': feature_columns if feature_columns else []
    })

if __name__ == '__main__':
    # Initialize the application with pre-trained model
    initialize_app()
    app.run(host='0.0.0.0', port=5000, debug=True)