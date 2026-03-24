<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=9B59D0&height=200&section=header&text=Shopper%20Insights&fontSize=70&fontColor=FFD700&animation=fadeIn&desc=K-Means%20Customer%20Segmentation&descAlignY=75" width="100%" />

# 🛍️ Shopper Insights with K-Means

<p align="center">
  <i>"Your customers have patterns. We make them visible."<br>A cutting-edge segmentation platform transforming raw data into actionable business intelligence.</i>
</p>

<img src="https://img.shields.io/badge/Python-3.8%2B-9B59D0?style=for-the-badge&logo=python&logoColor=white" alt="Python Version"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
<img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"/>

<br>
<br>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&pause=1000&color=FFD700&center=true&vCenter=true&width=435&lines=Analyze+Behaviors.;Cluster+Customers.;Predict+Segments.;Maximize+ROI." alt="Typing SVG" />

</div>

---

## ⚡ Core Features

<table align="center" width="100%">
  <tr>
    <td align="center" width="33%">
      <h3>🤖 Advanced K-Means</h3>
      <p>Unsupervised learning with automated optimal cluster detection (Elbow Method & Silhouette Analysis).</p>
    </td>
    <td align="center" width="33%">
      <h3>📊 Interactive EDA</h3>
      <p>Comprehensive data exploration with custom golden & purple themed Plotly visualizations.</p>
    </td>
    <td align="center" width="33%">
      <h3>⚡ Real-Time Classification</h3>
      <p>Instant segment prediction for new customer data via a RESTful Flask API.</p>
    </td>
  </tr>
</table>

---


## 🎯 Customer Segments Identified

Using the Mall Customer Segmentation dataset (analyzing Age, Gender, Annual Income, and Spending Score), the algorithm successfully identifies **5 distinct consumer profiles**:

| Segment | Profile | Behavior Pattern |
| :--- | :--- | :--- |
| 🟡 **Budget Shoppers** | Moderate Income | Price-conscious customers |
| 🟣 **Premium Customers** | High Income | High-spending luxury buyers |
| 🟡 **Conservative Spenders** | High Income | Cautious spending patterns |
| 🟣 **Young Spenders** | Moderate Income | Young demographic, high spending |
| 🟡 **Standard Customers** | Average Income | Average spending patterns |

---

## 🛠️ The Technology Stack

<table align="center">
  <tr>
    <td align="center"><b>Backend API</b></td>
    <td align="center"><b>Machine Learning</b></td>
    <td align="center"><b>Data Visualization</b></td>
    <td align="center"><b>Frontend UI</b></td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white" /><br>
      <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white" />
    </td>
    <td align="center">
      <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=flat&logo=scikit-learn&logoColor=white" /><br>
      <img src="https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white" /><br>
      <img src="https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white" />
    </td>
    <td align="center">
      <img src="https://img.shields.io/badge/Plotly-239120?style=flat&logo=plotly&logoColor=white" /><br>
      <img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=python&logoColor=white" /><br>
      <img src="https://img.shields.io/badge/Seaborn-4C72B0?style=flat&logo=python&logoColor=white" />
    </td>
    <td align="center">
      <img src="https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white" /><br>
      <img src="https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3&logoColor=white" /><br>
      <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black" />
    </td>
  </tr>
</table>

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
#### Step1: git clone <repository-url>
#### Step2: cd shopper-insights-kmeans
#### Step3: pip install flask flask-cors pandas numpy scikit-learn plotly matplotlib seaborn
#### Step4: python app.py
```

### 🔧 REST API Endpoints

#### Integrate our clustering model directly into your own applications:

##### GET /api/get_data_overview - Dataset overview and statistics

##### POST /api/perform_clustering - Execute clustering analysis with custom parameters

##### GET /api/get_cluster_analysis - Detailed cluster insights

##### POST /api/classify_customer - Submit age, income, and spending score to classify a new customer

### Example: Classify a Customer
#### POST /api/classify_customer
```bash
{
    "age": 25,
    "annual_income": 60,
    "spending_score": 80
}
```
