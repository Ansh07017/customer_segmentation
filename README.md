# Shopper Insights with K-Means

**Your customers have patterns. We make them visible.**

A comprehensive customer segmentation application using K-Means clustering with interactive visualizations and real-time customer classification.

## Features

- **Dark Theme Interface**: Professional purple and gold color scheme with enhanced visibility
- **Real Customer Data**: Uses authentic Mall Customer Segmentation dataset (200 customers)
- **Interactive EDA**: Comprehensive exploratory data analysis with dynamic visualizations
- **K-Means Clustering**: Configurable clustering with 3-5 segments and feature selection
- **Customer Classification**: Real-time classification with 99.5% accuracy using Random Forest
- **Business Insights**: Actionable marketing strategies for each customer segment

## Tech Stack

- **Backend**: Flask with RESTful API
- **Frontend**: Bootstrap 5 with Plotly.js visualizations
- **Machine Learning**: Scikit-learn (K-Means, Random Forest)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn

## Installation

### VS Code Setup

1. **Clone or download the project**
2. **Install dependencies**:
   ```bash
   pip install flask flask-cors pandas numpy scikit-learn plotly matplotlib seaborn
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Access the application**:
   Open browser to `http://localhost:5000`

### VS Code Debug Configuration

The project includes VS Code configuration files:
- `.vscode/launch.json` - Debug configuration for Flask app
- `.vscode/settings.json` - Python environment and formatting settings

## Usage

1. **Data Overview**: View dataset statistics and information
2. **EDA Analysis**: Explore customer demographics and spending patterns
3. **Customer Segmentation**: Configure clustering parameters and analyze segments
4. **Customer Classification**: Input customer data for real-time segment prediction

## API Endpoints

- `GET /` - Main dashboard
- `GET /api/get_data_overview` - Dataset overview and statistics
- `GET /api/get_eda` - Exploratory data analysis plots
- `POST /api/perform_clustering` - Perform K-Means clustering
- `GET /api/get_cluster_analysis` - Get clustering results
- `POST /api/classify_customer` - Classify new customer data

## Customer Segments

The application identifies distinct customer segments:
- **Young Medium Income Medium Spenders**
- **Middle-aged High Income High Spenders** 
- **Mature Medium Income Low Spenders**
- **Young Low Income Budget Conscious**
- **Premium High-Value Customers**

Each segment includes targeted marketing strategies and business insights.

## Data

Uses the Mall Customer Segmentation dataset with features:
- Customer ID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

## Development

The application automatically loads and processes the customer data on startup, trains the classification model, and provides immediate insights without manual data upload requirements.

## License

This project is for educational and commercial use in customer analytics and business intelligence applications.