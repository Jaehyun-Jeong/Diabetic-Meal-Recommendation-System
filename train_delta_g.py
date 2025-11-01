from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline


def load_data(
    features: list = [
        "carbs",
        "calories",
        "protein",
        "fat",
        "g0",
        "Age",
        "BMI",
        "Body weight ",
        "Height ",
        "has_diabetes",
        "Gender_F",
        "Gender_M",
    ]
):

    x_train = pd.read_csv("./dataset/x.clean.pruned.v3.csv")[features].to_numpy()
    y_train = pd.read_csv("./dataset/delta_g.clean.pruned.v3.csv").to_numpy().ravel()

    return x_train, y_train


param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.3, 0.5],
    "reg_alpha": [0, 0.01, 0.1, 1.0],  # L1 규제
    "reg_lambda": [0.1, 1.0, 10.0],  # L2 규제
}

features: list = [
    "carbs",
    "calories",
    "protein",
    "fat",
    "g0",
    "Age",
    "BMI",
    "Body weight ",
    "Height ",
    "has_diabetes",
    "Gender_F",
    "Gender_M",
]

x_train, y_train = load_data(features=features)

model = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    verbosity=1,
)

search = GridSearchCV(
    model,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=5,
    verbose=2,
    n_jobs=-1,
)
search.fit(x_train, y_train)

print("Best params:", search.best_params_)
print("Best MAE:", -search.best_score_)

# Lasso
# Best alpha: {'lasso__alpha': np.float64(0.14384498882876628), 'lasso__max_iter': 10000, 'lasso__selection': 'random', 'lasso__tol': 0.0001}
# Best score: 10.974423115352584

# RandomForestRegressor
# Best alpha: {'bootstrap': False, 'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
# Best score: 15.52786586618593

# CatBoostRegressor
# Best alpha: {'bagging_temperature': 0, 'depth': 4, 'iterations': 500, 'l2_leaf_reg': 3, 'learning_rate': 0.05, 'random_strength': 0.5}
# Best score: 10.149810708648548

# XGBoostRegressor
# Best alpha: {'colsample_bytree': 1.0, 'gamma': 0, 'learning_rate': 0.05, 'max_depth': 3, 'min_child_weight': 3, 'n_estimators': 500, 'reg_alpha': 0, 'reg_lambda': 1.5, 'subsample': 0.6}
# Best score: 10.109493567826284

# pruned data

# Best params: {'colsample_bytree': 0.7, 'gamma': 0, 'learning_rate': 0.01, 'max_depth': 10, 'min_child_weight': 1, 'n_estimators': 200, 'reg_alpha': 0.1, 'reg_lambda': 2.0, 'subsample': 0.8}
# Best MAE: 27.838292378055723

# pruned data v3
