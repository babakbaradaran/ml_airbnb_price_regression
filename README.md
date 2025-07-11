# 🏠 Airbnb Price Prediction - Exploratory Data Analysis (EDA)

This project explores Airbnb listing data to uncover patterns that influence pricing and build a predictive model. The goal is to help hosts set data-driven prices based on listing features.

---

## Project Goals

- Understand the structure and content of the Airbnb dataset
- Clean and preprocess price-related and relevant listing fields
- Perform deep exploratory data analysis (EDA) to reveal trends and outliers
- Visualize relationships between features and price (numeric & categorical)
- Engineer impactful features to improve model performance
- Train and evaluate regression models to predict listing prices
- Prepare a production-ready model for deployment

---

## Problem Statement

Airbnb hosts often rely on guesswork when pricing listings, leading to inconsistencies and missed revenue opportunities.  
This project uses historical listing data to predict nightly prices based on property features, reviews, and availability.

---

## What I Learned

- How to clean and convert currency values like `$1,200.00` to float
- Visual and statistical techniques for handling missing data
- Transforming skewed price distributions using log scaling
- Visualizing categorical and numeric features with `seaborn` and `matplotlib`
- Writing clean, modular, and production-friendly analysis code

---

## Dataset

- **Source:** [Inside Airbnb](http://insideairbnb.com/get-the-data.html)  
- **City:** Vancouver (You can choose your own city)  
- **File Used:** `listings.csv`  

Includes information on:
- Location, host features, pricing, availability, number of reviews, room type, etc.

---

## Notebooks

| Notebook           | Description |
|--------------------|-------------|
| `airbnb_eda.ipynb` | In-depth EDA, feature transformations, and data cleaning |

---

## Tools Used

- Python (pandas, numpy, matplotlib, seaborn)
- Jupyter Notebook / Google Colab
- GitHub for version control
- scikit-learn (in later stages)

---

## How to Run

### Option 1: Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/babakbaradaran/ml-projects/blob/main/01_airbnb_price_regression/notebooks/airbnb_eda.ipynb)

### Option 2: Run Locally

```bash
# Clone the repo
git clone https://github.com/babakbaradaran/ml-projects.git

# Navigate to project folder
cd ml-projects/01_airbnb_price_regression

# Install dependencies (if required)
pip install -r requirements.txt

# Launch the notebook
jupyter notebook notebooks/airbnb_eda.ipynb
