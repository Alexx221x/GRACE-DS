import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression

# Drop identifier columns that are not predictive
def drop_id_cols(X):
    return X.drop(columns=['CustomerId', 'Surname'], errors='ignore')

# Define categorical and numeric columns
categorical_cols = ['Geography', 'Gender']
numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

# Preprocessing: drop IDs, one-hot encode categoricals, scale numerics
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ],
    remainder='drop'
)

# Full pipeline
pipeline = Pipeline(steps=[
    ('drop_id', FunctionTransformer(drop_id_cols, validate=False)),
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', C=1.0,
                                       max_iter=1000, random_state=42))
])

# Fit on training data
X_train = train_df.drop(columns=[target_column])
y_train = train_df[target_column]
pipeline.fit(X_train, y_train)