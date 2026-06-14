import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

# Define columns
target_column = 'Exited'
categorical_cols = ['Geography', 'Gender']
numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
drop_cols = ['CustomerId', 'Surname']

# Custom transformer to drop unwanted columns
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.drop(columns=[c for c in self.columns_to_drop if c in X.columns])

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer([
    ('drop', ColumnDropper(drop_cols), [c for c in drop_cols if c in train_df.columns]),
    ('numeric', numeric_transformer, [c for c in numeric_cols if c in train_df.columns]),
    ('categorical', categorical_transformer, [c for c in categorical_cols if c in train_df.columns])
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000, C=0.1, class_weight='balanced'))
])

# Fit on training data
X_train = train_df.drop(columns=[target_column])
y_train = train_df[target_column]

pipeline.fit(X_train, y_train)

print("LogisticRegression pipeline fitted successfully.")
print("Classes seen:", pipeline.classes_)