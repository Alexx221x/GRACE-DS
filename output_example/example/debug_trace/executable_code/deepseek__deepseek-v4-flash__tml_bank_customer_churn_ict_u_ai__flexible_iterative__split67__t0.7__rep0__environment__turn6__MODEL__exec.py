import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

# Reproduce engineered features from earlier turn using a FunctionTransformer
def engineer_features(df):
    df = df.copy()
    # Age groups
    bins = [0, 30, 40, 50, 60, 100]
    labels = ['0-29', '30-39', '40-49', '50-59', '60+']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False).astype(str)
    # Balance-to-Salary ratio (avoid division by zero)
    df['BalanceSalaryRatio'] = np.where(df['EstimatedSalary'] > 0, df['Balance'] / df['EstimatedSalary'], 0.0)
    # Tenure-to-Age ratio
    df['TenureAgeRatio'] = np.where(df['Age'] > 0, df['Tenure'] / df['Age'], 0.0)
    # Product x Active interaction
    df['ProdsXActive'] = df['NumOfProducts'] * df['IsActiveMember']
    # Keep relevant columns (drop id and surname)
    cols = ['CreditScore', 'Geography', 'Gender', 'Age', 'Balance', 'NumOfProducts',
            'HasCrCard', 'IsActiveMember', 'AgeGroup', 'BalanceSalaryRatio',
            'TenureAgeRatio', 'ProdsXActive']
    return df[cols]

# Preprocessing: numeric scaling + categorical encoding
numeric_features = ['CreditScore', 'Age', 'Balance', 'NumOfProducts', 'HasCrCard',
                    'IsActiveMember', 'BalanceSalaryRatio', 'TenureAgeRatio', 'ProdsXActive']
categorical_features = ['Geography', 'Gender', 'AgeGroup']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

pipeline = Pipeline([
    ('engineer', FunctionTransformer(engineer_features)),
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42
    ))
])

# Fit on training data (excluding target)
X_train = train_df_original.drop(columns=[target_column])
y_train = train_df_original[target_column]
pipeline.fit(X_train, y_train)
print("GradientBoosting pipeline fitted successfully.")