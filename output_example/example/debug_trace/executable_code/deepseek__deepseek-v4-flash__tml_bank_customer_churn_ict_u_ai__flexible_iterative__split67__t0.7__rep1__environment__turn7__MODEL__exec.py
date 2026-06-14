import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier

target_column = 'Exited'

def preprocess(X):
    X = X.drop(columns=['CustomerId', 'Surname'], errors='ignore')
    X = X.copy()
    X['Age_IsActive'] = X['Age'] * X['IsActiveMember']
    X['Balance_to_Salary'] = X['Balance'] / (X['EstimatedSalary'] + 1)
    X['Tenure_Age'] = X['Tenure'] / (X['Age'] + 1)
    X['Age_bin'] = pd.cut(X['Age'], bins=[0, 30, 45, 60, 100], labels=[0,1,2,3]).astype(float)
    return X

numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
                'Age_IsActive', 'Balance_to_Salary', 'Tenure_Age', 'Age_bin']
cat_cols = ['Geography', 'Gender']

preprocessor = ColumnTransformer(transformers=[
    ('num', 'passthrough', numeric_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
])

pipeline = Pipeline([
    ('add_features', FunctionTransformer(preprocess, validate=False)),
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=300, max_depth=12,
                                   min_samples_leaf=3,
                                   class_weight='balanced',
                                   random_state=42))
])

X_train = train_df.drop(columns=[target_column])
y_train = train_df[target_column]
pipeline.fit(X_train, y_train)