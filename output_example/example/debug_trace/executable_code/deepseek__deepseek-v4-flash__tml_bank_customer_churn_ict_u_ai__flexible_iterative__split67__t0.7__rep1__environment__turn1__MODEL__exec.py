# --- Imports ---
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# --- Protect original data ---
target_column = 'Exited'
X_train = train_df.drop(columns=[target_column]).copy()
y_train = train_df[target_column].copy()

# --- Identify columns ---
id_cols = ['CustomerId', 'Surname']
cat_cols = ['Geography', 'Gender']
num_cols = [c for c in X_train.columns if c not in id_cols + cat_cols]

# --- Preprocessor ---
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

# --- Full pipeline with logistic regression (simple baseline) ---
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1))
])

# --- Fit ---
pipeline.fit(X_train, y_train)