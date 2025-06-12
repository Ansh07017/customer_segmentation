# 🚀 Deployment Guide - Shopper Insights with K-Means

## Local Development Setup

### Prerequisites
- Python 3.8 or higher
- Git (optional)
- VS Code (recommended)

### Quick Start (5 minutes)

1. **Download/Clone Project**
```bash
# If using Git
git clone <repository-url>
cd shopper-insights

# Or download and extract ZIP file
```

2. **Create Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install flask flask-cors pandas numpy scikit-learn plotly matplotlib seaborn
```

4. **Run Application**
```bash
python app.py
```

5. **Access Dashboard**
Open browser: `http://127.0.0.1:5000`

## VS Code Configuration

### Recommended Extensions
```
ms-python.python
ms-python.pylance
ms-toolsai.jupyter
ms-vscode.vscode-json
formulahendry.auto-rename-tag
```

### Debug Configuration
Press F5 to start debugging with the included launch configuration.

## Project Structure Summary

```
shopper-insights/
├── app.py                     # Main Flask application
├── run.py                     # Alternative entry point
├── README.md                  # Complete documentation
├── PRESENTATION_GUIDE.md      # PPT preparation guide
├── DEPLOYMENT_GUIDE.md        # This file
├── setup.py                   # Package configuration
├── .gitignore                 # Git ignore rules
├── .vscode/
│   ├── settings.json          # VS Code settings
│   └── launch.json           # Debug configuration
├── templates/
│   └── index.html            # Dashboard interface
├── utils/
│   ├── data_loader.py        # Data handling
│   ├── clustering.py         # K-Means implementation
│   ├── eda_functions.py      # Analysis functions
│   └── visualizations.py    # Chart generation
├── extracted_data/
│   └── Mall_Customers.csv    # Customer dataset
└── models/                   # Auto-generated ML models
    ├── kmeans_model.pkl      # Clustering model
    ├── scaler.pkl           # Data scaler
    └── rf_classifier.pkl    # Classification model
```

## Local Development Only

This application is configured for local development only on `127.0.0.1:5000`. 

### Running the Application
```bash
python app.py
```

The server will start on `http://127.0.0.1:5000` and is accessible locally only.

### VS Code Debug Configuration
Use F5 to start debugging with the included configuration that targets the local server.

## Troubleshooting

### Common Issues
1. **Port 5000 already in use**: Change port in app.py
2. **Module not found**: Ensure virtual environment is activated
3. **Permission errors**: Check file permissions on models/ directory
4. **Memory issues**: Reduce dataset size for testing

### Support Resources
- Check README.md for detailed documentation
- Review inline code comments
- Use VS Code debugger for step-through debugging