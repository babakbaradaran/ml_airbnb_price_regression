
# 🏠 Airbnb Price Prediction - Exploratory Data Analysis (EDA)

This project explores Airbnb listing data to identify patterns and relationships that influence pricing.

---

## Project Goals

- Understand the structure and content of the Airbnb dataset.
- Clean and transform price-related fields.
- Explore numerical and categorical variables.
- Identify insights useful for building a price prediction model.

---

## What I Learned

- How to safely clean currency values like `$1,200.00`.
- Handling missing data using visual and statistical methods.
- Log-transforming skewed price distributions for better modeling.
- Using `seaborn` to visualize relationships in categorical features.
- Writing clean, modular, and fail-safe analysis code.

---

## 📂 Project Structure

📁 01_airbnb_price_regression/
│
├── 📁 data/
│ └── 📁 raw/ # Original dataset (listings.csv)
│
├── 📁 notebooks/
│ └── airbnb_eda.ipynb # Jupyter notebook for analysis
│
└── README.md # Project overview


---

## 📦 Dataset

Data Source: [Inside Airbnb](http://insideairbnb.com/get-the-data.html)  
File used: `listings.csv` (I downloaded for Vancouver. You can download for your city of choice)

---

## How to Run

### Option 1: Google Colab  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/ml-projects/blob/main/01_airbnb_price_regression/notebooks/airbnb_eda.ipynb)

### Option 2: Local (VS Code or Jupyter)
```bash
cd ml-projects/01_airbnb_price_regression/
pip install -r requirements.txt  # if needed
jupyter notebook notebooks/airbnb_eda.ipynb
