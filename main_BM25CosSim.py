from typing import Dict, List
import numpy as np
import pandas as pd
from implicit.nearest_neighbours import bm25_weight

from DataLoader import (
    load_data,
    select_similar_features,
    split_train_val,
    create_y_target,
)
from model import BM25CosSim, save_model_normalizer, load_model_normalizer, predict_dict
from preprocess import Normalizer, idx_to_category

np.random.seed(42)

GOOD_MEAL_SCORE = 50.0  # 좋은 음식의 최소 기준 점수
BM25_BEST = {"K1": 3.02, "B": 1.99}  # 파라미터 튜닝으로 찾은 최적값
SIM_KEYS = [
    "Age",
    "Gender",
    "BMI",
    "Body weight ",
    "Height ",
]  # 유저의 유사도 계산을 위한 컬럼
VAL_SIZE = 10  # Validation에 사용할 환자의 수


# 유저의 유사도 계산, 식품군 추천을 위한 데이터 준비
def load_and_prepare_data(
    val_size: int = VAL_SIZE,
):

    df = load_data()
    # 'GOOD_MEAL_SCORE'를 넘긴 데이터만 사용
    df = df.loc[df["meal_score"] >= GOOD_MEAL_SCORE]
    # 환자 아이디
    patient_ids = df["patient_id"].unique()
    # 환자 아이디에서 validation에 사용할 환자의 아이디를 선택
    val_ids = np.random.choice(patient_ids, size=val_size, replace=False)

    # 식품군 데이터
    train_df, val_df = split_train_val(df=df, val_ids=val_ids)
    # 환자 데이터
    patient_train_df = select_similar_features(train_df, keys=SIM_KEYS)
    patient_val_df = select_similar_features(val_df, keys=SIM_KEYS)
    # 추천 데이터
    recommend_target = create_y_target(val_df)

    return train_df, patient_train_df, patient_val_df, recommend_target


# 데이터를 normalize
def normalize_data(train_df, val_df):
    normalizer = Normalizer()
    train_df = normalizer.fit_transform(train_df)
    val_df = normalizer.transform(val_df)
    return train_df, val_df, normalizer


# 모델 학습 및 정확도 출력
def train_and_evaluate(train_df, patient_val_df, recommend_target):
    model = BM25CosSim(
        K1=BM25_BEST["K1"],
        B=BM25_BEST["B"],
        sim_features=SIM_KEYS,
        key_x="patient_id",
        key_y="식품군분류",
    )
    model.fit(train_df)
    recommend_pred = model.predict(patient_val_df)
    recommend_pred = idx_to_category(recommend_pred)
    score = BM25CosSim.recall_at_K(recommend_pred, recommend_target)
    print(f"{score} at K1={BM25_BEST['K1']}, B={BM25_BEST['B']}")
    return model


def main():
    train_df, patient_train_df, patient_val_df, recommend_target = (
        load_and_prepare_data()
    )
    patient_train_df, patient_val_df, normalizer = normalize_data(
        patient_train_df, patient_val_df
    )

    model = train_and_evaluate(train_df, patient_val_df, recommend_target)
    model_path = "./saved_models/BM25CosSim_model.pkl"
    save_model_normalizer(model_path, model, normalizer)

    # 모델 불러오기
    model, normalizer = load_model_normalizer(model_path)
    # 가상의 환자 설정
    user_dict = {
        "patient_id": 0,
        "Age": 26,
        "Gender_M": 1.0,
        "Gender_F": 0.0,
        "BMI": 25.8,
        "Body weight ": 80,
        "Height ": 176,
    }

    # 가상의 환자에 대한 예측 결과
    recommend = predict_dict(user_dict, model, normalizer)
    print(recommend)


if __name__ == "__main__":
    main()
