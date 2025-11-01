import argparse
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

# 파이썬 실행 명령어로 데이터의 경로를 받음
parser = argparse.ArgumentParser()
parser.add_argument("--xpath", type=str)
parser.add_argument("--ypath", type=str)
args = parser.parse_args()

# 학습 데이터 불러오기
def load_data(x_path, y_path):

    x_train = pd.read_csv(x_path).to_numpy()
    y_train = pd.read_csv(y_path).to_numpy().ravel()

    return x_train, y_train

x_train, y_train = load_data(args.xpath, args.ypath)

# Grid Search로 찾은 최적의 파라미터
best_params = {'colsample_bytree': 0.7, 'gamma': 0, 'learning_rate': 0.01, 'max_depth': 10, 'min_child_weight': 1, 'n_estimators': 200, 'reg_alpha': 0.1, 'reg_lambda': 2.0, 'subsample': 0.8}

# 모델 불러오기
model = XGBRegressor(**best_params)

# 학습
'''
model.fit(x_train, y_train)
save_model(
    model=model,
    path="./saved_models/XGB_delta_g.pkl",
)
'''

# K-fold cross validation으로 검증
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, x_train, y_train, cv=kf, scoring='neg_mean_absolute_error')  # R^2 score

print("개별 Fold 성능:", scores)
print("평균 성능 (MAE):", np.mean(scores))
