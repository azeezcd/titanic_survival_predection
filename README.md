# Titanic - Machine Learning from Disaster

A binary classification pipeline built with Python and Scikit-Learn as part of an ML portfolio project.

## Workflow & Strategy
- **Data Preprocessing**: Imputed missing values (`Age`, `Embarked`) and dropped low-signal high-missingness attributes (`Cabin`).
- **Feature Engineering**: Created `FamilySize`, `IsAlone`, and extracted honorific titles (`Title`) from passenger names using regular expressions.
- **Model Selection & Tuning**: Used 5-Fold Stratified Cross-Validation to evaluate Logistic Regression vs. Random Forest Classifier.
- **Results**: Hyperparameter tuning via `GridSearchCV` improved cross-validation accuracy to **84.51%**.

## Repository Structure
- `main.ipynb`: Full EDA, feature engineering, cross-validation, and hyperparameter tuning notebook.
- `dataset/`: Training (`train.csv`) and test (`test.csv`) datasets.
- `submission.csv`: Final predictions formatted for Kaggle submission.
