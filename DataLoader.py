from typing import List
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy
from implicit.nearest_neighbours import bm25_weight
import implicit


def create_y_target(
    val_df: pd.DataFrame,
    key_x: str = 'patient_id',
    key_y: str = '식품군분류',
):

    val_df = val_df.groupby(
        [key_x, key_y],
        observed=False,
    ).size().unstack(fill_value=0)

    recommendations = {}
    for idx, value in enumerate(val_df.index):
        sorted_recommend = val_df.iloc[idx].sort_values().index.tolist()[::-1]
        recommendations[value] = sorted_recommend

    return recommendations


def split_train_val(
    df: pd.DataFrame,
    val_ids: np.ndarray
):

    # 선택된 환자들을 valid set, 나머지를 train set로 할당
    val_df = df[df['patient_id'].isin(val_ids)].copy()
    train_df = df[~df['patient_id'].isin(val_ids)].copy()

    return train_df, val_df

# EVERY COLUMN NAMES
# patient_id, meal_time, meal_type, carbs, protein, fat, fiber
# delta_g, g_max, GL, CHO_ratio, Protein_ratio, Fat_ratio
# 식품군분류, good_meal_label, meal_score
# Select features for cos similarity calculation among patients
def select_similar_features(
    df: pd.DataFrame,
    keys: list,
):

    new_df = df[['patient_id'] + keys].copy()

    # One hot all categorical columns
    categorical_cols = new_df.select_dtypes(include=['object', 'category']).columns
    new_df = pd.get_dummies(new_df, columns=categorical_cols, dtype=float)
    new_df = new_df.groupby('patient_id', observed=False).mean()

    return new_df


def binning(
    df: pd.DataFrame,
    keys: List[str],
):

    for key in keys:
        statistic = df[key].describe()

        # 25%, 50%, 75% quartiles
        q1 = statistic['25%']
        q2 = statistic['50%']
        q3 = statistic['75%']

        consumption_class = [f'<{q1}', f'{q1}~{q2}', f'{q2}~{q3}', f'>{q3}']    

        df.loc[df[key] < q1, f'{key}_consumption'] = consumption_class[0]
        df.loc[(df[key] >= q1) & (df[key] < q2), f'{key}_consumption'] = \
            consumption_class[1]
        df.loc[(df[key] >= q2) & (df[key] < q3), f'{key}_consumption'] = \
            consumption_class[2]
        df.loc[df[key] >= q3, f'{key}_consumption'] = consumption_class[3]

    return df


def load_data():

    meal_df = pd.read_csv("./dataset/evaluated_meals.csv")
    patient_df = pd.read_csv("./dataset/total_metrics.csv")  # patient information for similarity

    total_df = pd.concat(
        [meal_df, patient_df[patient_df.columns.difference(meal_df.columns)]],
        axis=1
    )

    return total_df


if __name__ == "__main__":

    total_df = load_data()
    print(total_df.head())
