# Titanic Survival Prediction

A machine learning project that predicts whether a passenger survived the Titanic disaster based on passenger data. Three classification models were trained, evaluated, and compared to identify the best performing approach.

---

## Problem Statement

The sinking of the Titanic in 1912 is one of the most infamous shipwrecks in history. This project uses passenger data — including age, sex, ticket class, and family size — to build a model that predicts survival outcomes.

This is a binary classification problem:
- **0** → Did not survive
- **1** → Survived

---

## Dataset

- **Source:** [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic)
- **Size:** 891 passengers, 12 features
- **Target variable:** `Survived`

| Feature | Description |
|---|---|
| Pclass | Passenger class (1st, 2nd, 3rd) |
| Sex | Gender of passenger |
| Age | Age in years |
| SibSp | Number of siblings/spouses aboard |
| Parch | Number of parents/children aboard |
| Fare | Ticket fare paid |
| Embarked | Port of embarkation (C, Q, S) |

---

## Approach

### 1. Exploratory Data Analysis
- Inspected shape, data types, and missing values
- Identified Cabin (77% missing), Age (20% missing), and Embarked (2 missing) as columns needing attention

### 2. Data Preprocessing
- Dropped `Cabin` column due to excessive missing values
- Filled missing `Age` values with mean age
- Dropped 2 rows with missing `Embarked` values
- Encoded `Sex` column — male → 1, female → 0
- Applied one-hot encoding to `Embarked` column using `get_dummies`
- Removed duplicate rows
- Applied `StandardScaler` to features for Logistic Regression

### 3. Feature Selection
Features used for training:
```
Sex, Age, Pclass, SibSp, Parch, Fare, Embarked_Q, Embarked_S
```

### 4. Model Training
Three classification models were trained and evaluated:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier (max_depth=4)

### 5. Evaluation Metrics
Each model was evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

---

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 80.6% | 73.4% | 75.4% | 74.4% |
| Decision Tree | 72.4% | 62.1% | 67.2% | 64.6% |
| **Random Forest** | **82.3%** | **83.7%** | 65.4% | 73.4% |

### Confusion Matrix — Random Forest (Best Model)

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 170 (TN) | 14 (FP) |
| **Actual 1** | 38 (FN) | 72 (TP) |

---

## Key Findings

- **Random Forest** achieved the highest accuracy (82.3%) and precision (83.7%), meaning when it predicted survival it was correct most of the time
- **Logistic Regression** achieved better recall (75.4%), meaning it caught more actual survivors — preferable in scenarios where missing a survivor is more costly than a false alarm
- **Decision Tree** underperformed both models, likely due to overfitting on the training data without hyperparameter tuning
- **Sex, Pclass, and Age** were the most important features in predicting survival
- Hyperparameter tuning of `max_depth` on Random Forest showed that depth of 4 and 5 produced identical results, suggesting the model converged at that complexity level

---

## Tech Stack

- **Language:** Python 3
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Environment:** Google Colab

---

## How to Run

1. Clone this repository
```bash
git clone https://github.com/Sunny-sketchs/titanic-survival-prediction.git
cd titanic-survival-prediction
```

2. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

3. Open the notebook
```bash
jupyter notebook Titanic_project.ipynb
```
Or upload directly to [Google Colab](https://colab.research.google.com/)

4. Run all cells in order

---

## Project Structure

```
titanic-survival-prediction/
│
├── Titanic_project.ipynb    # Main notebook with full analysis
├── Titanic-Dataset.csv      # Dataset from Kaggle
└── README.md                # Project documentation
```

---

## Future Improvements

- Feature engineering — create FamilySize column (SibSp + Parch + 1)
- Try XGBoost and cross validation for better performance
- Hyperparameter tuning using GridSearchCV
- Deploy model as a web app using FastAPI and Streamlit

---

## Author

**Sunny Bhatkar**  
[GitHub](https://github.com/Sunny-sketchs) • [LinkedIn](https://www.linkedin.com/in/sunny-bhatkar-75793828a)
