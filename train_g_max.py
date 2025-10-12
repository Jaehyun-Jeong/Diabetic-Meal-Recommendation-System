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

def load_data():

    x_train = pd.read_csv("./dataset/x.clean.pruned.v2.csv").to_numpy()
    y_train = pd.read_csv("./dataset/g_max.clean.pruned.v2.csv").to_numpy().ravel()

    return x_train, y_train

x_train, y_train = load_data()

param_grid = {
    'n_estimators': [200, 400, 600],             # 충분히 깊은 학습
    'learning_rate': [0.01, 0.05, 0.1],          # 느리지만 정교한 학습도 포함
    'max_depth': [4, 6, 8, 10],                  # 깊이 다양화 (복잡도 조절)
    'min_child_weight': [1, 3, 5],               # 과적합 방지
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],           # 트리마다 데이터 샘플링 비율
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],    # 피처 샘플링 다양화
    'gamma': [0],                      # 노드 분할 민감도 조절
    'reg_alpha': [0],                 # L1 정규화
    'reg_lambda': [1.0, 1.5, 2.0]                # L2 정규화
}

# Best alpha: {'colsample_bytree': 1.0, 'gamma': 0, 'learning_rate': 0.05, 'max_depth': 3, 'min_child_weight': 3, 'n_estimators': 500, 'reg_alpha': 0, 'reg_lambda': 1.5, 'subsample': 0.6}

model = XGBRegressor(
    objective='reg:squarederror',
    n_jobs=-1,
    verbosity=0
)

search = GridSearchCV(
    model,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',
    cv=5,
    verbose=2,
    n_jobs=1
)
search.fit(x_train, y_train)

print("Best params:", search.best_params_)
print("Best MAE:", -search.best_score_)

# x.clean.pruned.v2.csv

# Lasso
# Best params: {'lasso__alpha': np.float64(0.38566204211634725), 'lasso__selection': 'random', 'lasso__tol': 0.01}
# Best MAE: 30.726041059942947
