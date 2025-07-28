from typing import Dict, List
import numpy as np
import pandas as pd
from implicit.nearest_neighbours import bm25_weight
from DataLoader import load_data, select_similar_features, split_train_val, create_y_target
from model import BM25CosSim, save_model_normalizer, load_model_normalizer, predict_dict
from preprocess import Normalizer, idx_to_category

np.random.seed(42)

GOOD_MEAL_SCORE = 50.0
BM25_BEST = {'K1': 3.02, 'B': 1.99}
SIM_KEYS = ['Age', 'Gender', 'BMI', 'Body weight ', 'Height ']
VAL_SIZE = 10

def load_and_prepare_data(
    val_size: int = VAL_SIZE,
):

    df = load_data()
    df = df.loc[df['meal_score'] >= GOOD_MEAL_SCORE]
    patient_ids = df['patient_id'].unique()
    val_ids = np.random.choice(patient_ids, size=val_size, replace=False)

    train_df, val_df = split_train_val(df=df, val_ids=val_ids)
    patient_train_df = select_similar_features(train_df, keys=SIM_KEYS)
    patient_val_df = select_similar_features(val_df, keys=SIM_KEYS)
    recommend_target = create_y_target(val_df)

    return train_df, patient_train_df, patient_val_df, recommend_target


def normalize_data(train_df, val_df):
    normalizer = Normalizer()
    train_df = normalizer.fit_transform(train_df)
    val_df = normalizer.transform(val_df)
    return train_df, val_df, normalizer


def train_and_evaluate(train_df, patient_val_df, recommend_target):
    model = BM25CosSim(
        K1=BM25_BEST['K1'], B=BM25_BEST['B'],
        sim_features=SIM_KEYS,
        key_x='patient_id', key_y='식품군분류'
    )
    model.fit(train_df)
    recommend_pred = model.predict(patient_val_df)
    recommend_pred = idx_to_category(recommend_pred)
    score = BM25CosSim.recall_at_K(recommend_pred, recommend_target)
    print(f"{score} at K1={BM25_BEST['K1']}, B={BM25_BEST['B']}")
    return model


def main():
    train_df, patient_train_df, patient_val_df, recommend_target = load_and_prepare_data()
    patient_train_df, patient_val_df, normalizer = normalize_data(patient_train_df, patient_val_df)

    model = train_and_evaluate(train_df, patient_val_df, recommend_target)
    model_path = "./saved_models/BM25CosSim_model.pkl"
    save_model_normalizer(
        model_path,
        model,
        normalizer
    )

    # load test
    model, normalizer = load_model_normalizer(model_path)
    user_dict = {
        'patient_id': 0,
        'Age': 26,
        'Gender_M': 1.,
        'Gender_F': 0.,
        'BMI': 25.8,
        'Body weight ': 80,
        'Height ': 176,
    }

    recommend = predict_dict(user_dict, model, normalizer)

    print(recommend)

if __name__ == "__main__":
    main()
