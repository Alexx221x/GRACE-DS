import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Identify feature types
target_column = 'Exited'
id_cols = ['CustomerId', 'Surname']
categorical_cols = ['Geography', 'Gender']
numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

# Preprocessing: drop id cols, one-hot encode categoricals, scale numerics
preprocessor = ColumnTransformer([
    ('drop', 'drop', id_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols),
    ('num', StandardScaler(), numeric_cols)
])

# RandomForest classifier as a different model family
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced')

pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', rf)
])

# Fit on training data
X_train = train_df_original.drop(columns=[target_column])
y_train = train_df_original[target_column]
pipeline.fit(X_train, y_train)

print("RandomForest pipeline fitted successfully.")
print(f"Classes seen: {pipeline.classes_}")