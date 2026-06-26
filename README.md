# Automobile-Price-Regression
A machine learning project that predicts automobile prices using regression and classification techniques on a real-world automobile dataset.

---

#  Project Overview
 
This project analyzes a dataset of automobile specifications to:
- Predict the **exact price** of a car using regression models (Linear Regression & Random Forest)
- Classify cars into **price categories** (Low / Medium / High) using Logistic Regression
Two train/test splits (80/20 and 70/30) are compared to evaluate model stability and performance.
 
---

## Dataset
 
The dataset contains **26 features** describing automobile characteristics:
 
| Category | Features |
|---|---|
| Identity | symboling, make |
| Body | fuel-type, aspiration, num-of-doors, body-style, drive-wheels, engine-location |
| Dimensions | wheel-base, length, width, height, curb-weight |
| Engine | engine-type, num-of-cylinders, engine-size, fuel-system, bore, stroke, compression-ratio, horsepower, peak-rpm |
| Fuel Economy | city-mpg, highway-mpg |
| Target | **price** |

---

##  Data Preprocessing
 
- Replaced `"?"` placeholder values with `NaN`
- Imputed missing numeric values with **column mean**
- Imputed missing categorical values with **column mode**
- Converted columns to appropriate data types
- Treated outliers in `price`, `horsepower`, and `engine-size` using **IQR-based capping (Winsorization)**

---

### Feature Engineering
- Created `highway-L/100km` and `city-L/100km` from MPG values
- Applied **binning** on horsepower → Low / Medium / High
- Applied **min-max normalization** on engine size

---

#  Exploratory Data Analysis
 
Visualizations created:
- Histogram of Price distribution
- Histogram of Horsepower distribution
- Boxplot: Price vs Body Style
- Correlation Heatmap (numeric features)
**Key findings:**
- `engine-size` and `horsepower` have a strong positive correlation with price
- Fuel consumption variables (L/100km) also relate significantly to price

---

##  Models
 
### Regression (Predicting Exact Price)
 
Features used: `engine-size`, `horsepower`, `city-L/100km`, `highway-L/100km`
 
| Model | Split | R² Score |
|---|---|---|
| Linear Regression | 80/20 | — |
| Random Forest | 80/20 | ✅ Best |
| Linear Regression | 70/30 | — |
| Random Forest | 70/30 | — |
 
> **Random Forest (80/20) achieved the best performance** across all models and splits, capturing non-linear relationships that Linear Regression could not model.
 
Metrics reported: **R², MAE, MSE**

Metrics reported: **R², MAE, MSE**
 
### Classification (Predicting Price Category)
 
Cars were categorized into 3 equal-frequency bins using `pd.qcut`:
- Low Price
- Medium Price
- High Price
**Model:** Logistic Regression (max_iter=500)
 
**Evaluation:** Accuracy Score + Confusion Matrix
 
---

## Tech Stack
 
- **Python 3**
- **pandas** — data manipulation
- **numpy** — numerical operations
- **matplotlib / seaborn** — visualizations
- **scikit-learn** — machine learning models and evaluation

---

## Results Summary
 
- **Random Forest Regressor** outperformed Linear Regression on both splits due to its ability to capture non-linear patterns in the data
- **Logistic Regression** successfully classified most vehicles into the correct price category, showing that vehicle specs carry strong signals for price range prediction
- The **80/20 split** yielded the best Random Forest results overall

---
