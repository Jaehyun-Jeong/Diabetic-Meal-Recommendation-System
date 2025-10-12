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

from model import save_model

def load_data():

    x_train = pd.read_csv("./dataset/x.clean.pruned.v2.csv").to_numpy()
    y_train = pd.read_csv("./dataset/g_max.clean.pruned.v2.csv").to_numpy().ravel()

    return x_train, y_train

x_train, y_train = load_data()

best_params = {'colsample_bytree': 0.7, 'gamma': 0, 'learning_rate': 0.01, 'max_depth': 10, 'min_child_weight': 1, 'n_estimators': 200, 'reg_alpha': 0.1, 'reg_lambda': 2.0, 'subsample': 0.8}

model = XGBRegressor(**best_params)
model.fit(x_train, y_train)
save_model(
    model=model,
    path="./saved_models/XGB_g_max.pkl",
)

'''
# K-fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, x_train, y_train, cv=kf, scoring='neg_mean_absolute_error', verbose=2)

print("개별 Fold 성능:", scores)
print("평균 성능 (MAE):", np.mean(scores))
'''
