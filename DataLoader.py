from typing import List
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy
from implicit.nearest_neighbours import bm25_weight
import implicit


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


def load_data(
    file_path: str,
):

    return pd.read_csv(file_path)


if __name__ == "__main__":

    total_df = load_data("./dataset/total_metrics.csv")

    keys = ['carbs', 'protein', 'fat']
    df = binning(total_df, keys)
    size_val = 10  # use 10 patients to validate

    # ========================================
    # Split train, val
    # ========================================

    patient_ids = df['patient_id'].unique()
    np.random.seed(42)
    val_ids = np.random.choice(unique_ids, size=size_val, replace=False)

    # 선택된 환자들을 valid set, 나머지를 train set로 할당
    val_df = df[df['patient_id'].isin(val_ids)].copy()
    train_df = df[~df['patient_id'].isin(val_ids)].copy()

    # ========================================

    patient_carb_mat = train_df.groupby(
        ['patient_id', 'carbs_consumption']
    ).size().unstack(fill_value=0)
    patient_carb_mat = bm25_weight(
        patient_carb_mat,
        K1=1.5,
        B=0.75,
    )
    patient_carb_mat = pd.DataFrame.sparse.from_spmatrix(patient_carb_mat)
    patient_carb_mat.index = train_df.groupby(
        ['patient_id', 'carbs_consumption']
    ).size().unstack(fill_value=0).index
    patient_carb_mat.columns= train_df.groupby(
        ['patient_id', 'carbs_consumption']
    ).size().unstack(fill_value=0).columns

    # FIX: Use other features to calculate similarity among patients
    patient_similarity = pdist(patient_carb_mat, metric='cosine')
    patient_similarity = 1 - squareform(patient_similarity)

    patient_score_pred = patient_similarity.dot(
        (patient_carb_mat) / np.array([np.abs(patient_similarity).sum(axis=1)]).T
    )

    print(patient_score_pred)
    print(patient_score_pred.shape)
    raise ValueError("test")

    # Recommend
    recommendations = {}
    for idx, patient in enumerate(patient_carb_mat.index):
        sorted_indicies = patient_score_pred[idx].argsort()[::-1]
        sorted_recommend = [
            amount for amount in patient_carb_mat.columns[sorted_indicies]
        ]

        recommendations[patient] = sorted_recommend

    print(recommendations)
