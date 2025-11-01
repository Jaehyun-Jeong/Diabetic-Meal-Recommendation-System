from itertools import product
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import scipy
from implicit.nearest_neighbours import bm25_weight
import implicit
from tqdm import tqdm

from DataLoader import (
    load_data,
    select_similar_features,
    split_train_val,
    create_y_target,
)
from model import BM25CosSim, Normalizer

np.random.seed(42)
# 가장 좋은 파라미터
BM25_BEST = {
    "K1": 3.02,
    "B": 1.99,
}

# 전체 식품군 리스트
FOOD_CATEGORIES = [
    "과일군",
    "곡류군",
    "혼합식품",
    "어육류군",
    "우유군",
    "채소군",
    "지방군",
]
# 좋은 식사의 기준점수
GOOD_MEAL_SCORE = 50.0
df = load_data()
df = df.loc[df["meal_score"] >= GOOD_MEAL_SCORE]

# 환자를 train, validation으로 나누는 과정
size_val = 10  # use 10 patients to validate
patient_ids = df["patient_id"].unique()
val_ids = np.random.choice(patient_ids, size=size_val, replace=False)
train_df, val_df = split_train_val(
    df=df,
    val_ids=val_ids,
)
train_df["식품군분류"] = pd.Categorical(
    train_df["식품군분류"], categories=FOOD_CATEGORIES
)

# every column names
# patient_id, meal_time, meal_type, carbs, protein, fat, fiber
# delta_g, g_max, gl, cho_ratio, protein_ratio, fat_ratio
# 식품군분류, good_meal_label, meal_score
# Age, Gender, BMI, Body weight, Height
# 환자 유사도 측정을 위해 사용하는 컬럼
keys = ["Age", "Gender", "BMI", "Body weight ", "Height "]
# 환자 학습 데이터
patient_train_df = select_similar_features(
    train_df,
    keys=keys,
)
# 환자 추천 데이터
recommend_target = create_y_target(val_df)
# 환자 검증 데이터
patient_val_df = select_similar_features(
    val_df,
    keys=keys,
)

# Normalize
mean = patient_train_df.mean()
std = patient_train_df.std()
patient_train_df = (patient_train_df - mean) / std
patient_val_df = (patient_val_df - mean) / std

# 학습 진행
model = BM25CosSim(K1=BM25_BEST["K1"], B=BM25_BEST["B"])

model.fit(
    train_df=train_df,
    key_x="patient_id",
    key_y="식품군분류",
    sim_df=patient_train_df,
)

# 검증
recommend_pred = model.predict(patient_val_df)
# recall@K 로 점수 측정
score = BM25CosSim.recall_at_K(recommend_pred, recommend_target)

print(score)
