# 🏙️ Dubai Property Price Prediction

<div align="center">

![Dubai Real Estate](https://img.shields.io/badge/Dubai%20Property-Price%20Prediction-8b5cf6?style=for-the-badge&logo=python&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-a855f7?style=for-the-badge&logo=python&logoColor=white)
![Regression](https://img.shields.io/badge/Machine%20Learning-Regression-c084fc?style=for-the-badge&logo=scikit-learn&logoColor=white)

### *Predicting Dubai property prices using machine learning*

[Overview](#-overview) • [Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Model Performance](#-model-performance) • [Contributing](#-contributing)

</div>

---

## 📊 Overview

This project implements a **machine learning regression pipeline** to predict property prices in Dubai based on real estate features. Using data preprocessing, exploratory data analysis (EDA), feature engineering, and Linear Regression, the model estimates housing prices with strong predictive performance.

The dataset contains **1,905 property records** with multiple features such as area, bedrooms, bathrooms, and neighborhood information.

### 🎯 Project Goals

- Perform comprehensive exploratory data analysis on property data
- Visualize feature relationships and correlations
- Build a regression model using Linear Regression
- Evaluate model performance using multiple regression metrics
- Understand key factors influencing Dubai property prices

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🔍 Data Analysis
- Dataset with 1,905 property listings
- Statistical summary of numerical features
- Correlation heatmap analysis
- Price distribution visualization
- Neighborhood-wise price comparison
- Area vs Price relationship plots

</td>
<td width="50%">

### 🤖 Machine Learning
- Linear Regression implementation
- Feature scaling using StandardScaler
- Train-test split (80/20)
- Model evaluation using MAE, MSE, RMSE
- R² Score: **0.717**

</td>
</tr>
</table>

---

## 🛠️ Tech Stack

<div align="center">

| Library | Purpose | Version |
|---------|---------|---------|
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) | Data manipulation and analysis | Latest |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | Numerical computations | Latest |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat&logo=python&logoColor=white) | Data visualization | Latest |
| ![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat&logo=python&logoColor=white) | Statistical visualizations | Latest |
| ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | Machine learning algorithms | Latest |
| ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) | Interactive development | Latest |

</div>

---

## 📥 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/Dubai-Property-Price-Prediction.git

# Navigate to project directory
cd Dubai-Property-Price-Prediction

# Install required dependencies
pip install -r requirements.txt
```

### Requirements File
```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

## 🚀 Usage

### Running the Jupyter Notebook

```bash
# Start Jupyter Notebook
jupyter notebook

# Open the main notebook file
# Navigate to Dubai_Property_Price_Prediction.ipynb
```

### Quick Start Code

```python
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load dataset
df = pd.read_csv('properties_data.csv')

# Define features and target
X = df.drop('price', axis=1)
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"R² Score: {r2:.3f}")
```

---

## 📈 Model Performance

<div align="center">

### Linear Regression Results

| Metric | Value |
|--------|-------|
| **MAE** | Evaluated |
| **MSE** | Evaluated |
| **RMSE** | Evaluated |
| **R² Score** | 0.717 |

### 🎯 Model explains approximately 71.7% of variance in property prices.

</div>

### Key Insights

- **Strong Relationship**: Property area has the strongest correlation with price
- **Neighborhood Impact**: Certain neighborhoods significantly impact property value
- **Feature Scaling**: StandardScaler improves regression performance
- **Multicollinearity**: Moderate multicollinearity observed in some numerical features

---

## 📁 Project Structure

```
Dubai-Property-Price-Prediction/
│
├── properties_data.csv                      # Dataset
│
├── Dubai_Property_Price_Prediction.ipynb    # Main Jupyter notebook
│
├── requirements.txt                         # Python dependencies
├── README.md                                # Project documentation
└── LICENSE
```

---

## 🎨 Visualizations

The project includes comprehensive visualizations:

- 📊 **Correlation Heatmap** - Shows relationships between all numerical features
- 📈 **Price Distribution Plot** - Distribution of property prices across the dataset
- 📉 **Area vs Price Scatter Plot** - Relationship between property size and price
- 🏘️ **Neighborhood-wise Price Analysis** - Average prices compared across neighborhoods
- 📦 **Statistical Summary** - Complete descriptive statistics for all features

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. 🍴 Fork the repository
2. 🔨 Create a new branch (`git checkout -b feature/improvement`)
3. 💾 Commit your changes (`git commit -am 'Add new feature'`)
4. 📤 Push to the branch (`git push origin feature/improvement`)
5. 🔃 Create a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Parisha Sharma**

- GitHub: [@parisha-sharma](https://github.com/parisha-sharma)
- LinkedIn: [parishasharma15](https://www.linkedin.com/in/parishasharma15)

---

## 🌟 Acknowledgments

- Dataset source: Kaggle
- Inspired by real-world real estate analytics
- Built as part of Data Science learning journey

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

![Made with Love](https://img.shields.io/badge/Made%20with-❤️-8b5cf6?style=for-the-badge)
![Data Science](https://img.shields.io/badge/Data%20Science-Learning-a855f7?style=for-the-badge)

**Happy Learning! 🚀**

</div>
