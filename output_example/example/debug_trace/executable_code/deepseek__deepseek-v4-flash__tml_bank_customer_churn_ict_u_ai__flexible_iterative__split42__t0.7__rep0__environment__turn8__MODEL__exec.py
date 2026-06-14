import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier

target_column = 'Exited'

# Separate features from target
X_train = train_df.drop(columns=[target_column])
y_train = train_df[target_column]

# Identify column types
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
# Drop CustomerId from numeric (high cardinality identifier)
numeric_cols = [c for c in numeric_cols if c not in ['CustomerId']]
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
# Also treat HasCrCard, IsActiveMember as categorical low-cardinality? They are numeric 0/1, keep as numeric.

# Add engineered features: balance-to-salary ratio, tenure-to-age ratio, product count squared, age squared
def add_features(X):
    X = X.copy()
    X['BalanceToSalary'] = X['Balance'] / (X['EstimatedSalary'] + 1e-8)
    X['TenureToAge'] = X['Tenure'] / (X['Age'] + 1e-8)
    X['NumOfProducts_sq'] = X['NumOfProducts'] ** 2
    X['Age_sq'] = X['Age'] ** 2
    # Interaction: IsActiveMember * NumOfProducts
    X['ActiveXProducts'] = X['IsActiveMember'] * X['NumOfProducts']
    return X

# Preprocessing: numeric pipeline (impute median, scale), categorical pipeline (impute most_frequent, one-hot)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Feature engineering step
feature_engineering = FunctionTransformer(add_features, validate=False)

# Determine post-feature-engineering column types (simplify: all numeric after engineering + OHE)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='drop'
)

pipeline = Pipeline(steps=[
    ('features', feature_engineering),
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean', n_jobs=-1))
])

pipeline.fit(X_train, y_train)
print("KNN pipeline fitted.")
print(f"Classes: {pipeline.classes_}")