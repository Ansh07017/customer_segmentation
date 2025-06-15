<<<<<<< HEAD
# Shopper Insights with K-Means

*"Your customers have patterns. We make them visible."*

A cutting-edge customer segmentation platform that transforms raw data into actionable business intelligence through advanced machine learning techniques.

## 🚀 Features

### Core Functionality
- **Advanced K-Means Clustering**: Unsupervised learning for customer segmentation
- **Interactive Data Exploration**: Comprehensive EDA with golden & purple themed visualizations
- **Real-time Classification**: Instant customer segment prediction
- **Statistical Analysis**: Detailed cluster insights and comparisons
- **Optimal Cluster Detection**: Automated elbow method and silhouette analysis

### Visualizations
- Correlation matrices with enhanced color schemes
- Interactive scatter plots and cluster visualizations
- Distribution analysis charts
- Radar charts for cluster profiling
- Feature importance analysis

## 🛠️ Technology Stack

- **Backend**: Python Flask with RESTful API
- **Machine Learning**: scikit-learn K-Means clustering
- **Data Processing**: Pandas, NumPy
- **Visualizations**: Plotly (interactive), Matplotlib, Seaborn
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Development**: VS Code ready with full configuration

## 📊 Dataset

Uses the Mall Customer Segmentation dataset featuring:
- Customer demographics (Age, Gender)
- Financial data (Annual Income)
- Behavioral metrics (Spending Score)
- 200 customer records for analysis

## 🎯 Customer Segments Identified

1. **Budget Shoppers** - Price-conscious customers with moderate income
2. **Premium Customers** - High-income, high-spending luxury buyers
3. **Conservative Spenders** - High-income but cautious spending patterns
4. **Young Spenders** - Young customers with moderate income but high spending
5. **Standard Customers** - Average income and spending patterns

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- VS Code (recommended)

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd shopper-insights-kmeans
   ```

2. **Install dependencies** (handled automatically in Replit)
   ```bash
   pip install flask flask-cors pandas numpy scikit-learn plotly matplotlib seaborn
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the application**
   - Open your browser to `http://localhost:5000`
   - The app automatically loads sample data and trains the model

### VS Code Development

The project includes full VS Code configuration:

- **Debugging**: Press F5 to start debugging the Flask app
- **Tasks**: Use Ctrl+Shift+P → "Tasks: Run Task" for common operations
- **Extensions**: Recommended extensions will be suggested automatically
- **Formatting**: Code formatting on save with Black
- **Linting**: Integrated Python linting with Pylint and Flake8

## 📁 Project Structure

```
shopper-insights-kmeans/
├── .vscode/                 # VS Code configuration
│   ├── settings.json        # Editor settings and Python config
│   ├── launch.json          # Debug configurations
│   ├── tasks.json           # Build and run tasks
│   └── extensions.json      # Recommended extensions
├── utils/                   # Core utilities
│   ├── clustering.py        # K-Means implementation
│   ├── data_loader.py       # Data processing
│   ├── eda_functions.py     # Exploratory data analysis
│   └── visualizations.py   # Plotly visualizations
├── templates/               # HTML templates
│   └── index.html           # Main application interface
├── models/                  # Trained models storage
├── extracted_data/          # Dataset files
│   └── Mall_Customers.csv   # Sample dataset
├── app.py                   # Flask application entry point
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## 🔧 API Endpoints

- `GET /` - Main application interface
- `GET /api/get_data_overview` - Dataset overview and statistics
- `GET /api/get_eda` - Exploratory data analysis results
- `POST /api/perform_clustering` - Execute clustering analysis
- `GET /api/get_cluster_analysis` - Detailed cluster insights
- `GET /api/get_optimal_clusters` - Optimal cluster number analysis
- `POST /api/classify_customer` - Classify new customer data
- `GET /api/model_status` - Model training status

## 🎨 Design Theme

The application features a sophisticated golden and purple color scheme:
- **Primary Gold**: #FFD700 - For highlights and accents
- **Primary Purple**: #9b59d0 - For interactive elements
- **Dark Background**: rgba(15, 15, 25, 1.0) - For enhanced contrast
- **Clean Typography**: Arial Black for readability

## 🔍 Usage Examples

### Basic Customer Classification
```python
# Example customer data
customer_data = {
    "age": 25,
    "annual_income": 60,
    "spending_score": 80
}

# Submit via the web interface or API
POST /api/classify_customer
```

### Custom Clustering Analysis
```python
# Perform clustering with custom parameters
clustering_params = {
    "features": ["Age", "Annual Income (k$)", "Spending Score (1-100)"],
    "n_clusters": 5,
    "scale_features": true
}

POST /api/perform_clustering
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper testing
4. Ensure code formatting with Black
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Mall Customer Segmentation dataset from Kaggle
- scikit-learn community for machine learning tools
- Plotly team for interactive visualizations

---

*Built with ❤️ for data-driven business insights*
=======
# customer_segmentation
>>>>>>> e6a7532ed6ce4143dc373de7cf5b31042537dcd9
