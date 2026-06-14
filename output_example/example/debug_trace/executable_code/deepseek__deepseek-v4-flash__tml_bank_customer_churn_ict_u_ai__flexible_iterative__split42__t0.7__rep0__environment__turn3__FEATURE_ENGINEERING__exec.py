import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

# Define feature creation that can be applied to raw feature rows
# Drop identifier columns and high-cardinality string, encode categoricals, keep numeric

drop_cols = ['CustomerId', 'Surname']
cat_cols = ['Geography', 'Gender']
num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

# Drop columns not useful for modelling
def drop_columns(X):
    return X.drop(columns=drop_cols, errors='ignore')

drop_step = FunctionTransformer(drop_columns, validate=False)

# Column transformer for preprocessing
column_prep = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_cols)
    ],
    verbose_feature_names_out=False
)

preprocessor = Pipeline(steps=[
    ('drop', drop_step),
    ('prep', column_prep)
])

# Fit on training data to inspect shape and feature names
X_train_demo = train_df_original.drop(columns=[target_column])
preprocessor.fit(X_train_demo)
transformed = preprocessor.transform(X_train_demo)
feature_names = preprocessor.get_feature_names_out()
print(f"Preprocessor output shape: {transformed.shape}")
print(f"Feature names: {feature_names.tolist()}")
print(f"Numeric columns used: {num_cols}")