import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import scipy
from implicit.nearest_neighbours import bm25_weight
import implicit
from sklearn.metrics.pairwise import cosine_similarity


class BM25CosSim():

    # B: [0, 1]. increase around 0.1, optimal [0.3, 0.9]
    # K1 [0, 3], increase around 0.1 to 0.2, optimal [0.5, 2.0]

    def __init__(
        self,
        K1: float = 3.95,
        B: float = 0.2,
    ):

        self.bm25_weight = None
        self.base_df = None

    @staticmethod 
    def __bm25_weight(
        train_df: pd.DataFrame,
        key_x: str = 'patient_id',
        key_y: str = '식품군분류',
    ):

        mat = train_df.groupby(
            [key_x, key_y],
            observed=False,
        ).size().unstack(fill_value=0)
        mat = bm25_weight(
            mat,
            K1=1.5,
            B=0.75,
        )
        mat = pd.DataFrame.sparse.from_spmatrix(mat)
        mat.index = train_df.groupby(
            [key_x, key_y],
            observed=False,
        ).size().unstack(fill_value=0).index
        mat.columns = train_df.groupby(
            [key_x, key_y],
            observed=False,
        ).size().unstack(fill_value=0).columns

        return mat

    def fit(
        self,
        train_df: pd.DataFrame,  # Matrix for BM25
        sim_df: pd.DataFrame,  # Matrix for similarity
        key_x: str = 'patient_id',  # BM25 row
        key_y: str = '식품군분류',  # BM25 col (categorical)
    ):

        self.bm25_weight = self.__bm25_weight(train_df, key_x, key_y)
        self.base_df = sim_df

    def predict(
        self,
        y: pd.DataFrame,
    ):

        similarity = cosine_similarity(self.base_df, y)

        score_pred = similarity.T.dot(
            (self.bm25_weight) / np.array([np.abs(similarity).sum(axis=1)]).T
        )

        # Recommend
        recommendations = {}
        for idx, value in enumerate(y.index):
            sorted_indicies = score_pred[idx].argsort()[::-1]
            sorted_recommend = [
                amount for amount in self.bm25_weight.columns[sorted_indicies]
            ]

            recommendations[value] = sorted_recommend

        return recommendations

    @staticmethod
    def recallK(
        y_pred: dict,
        y: dict,
        K: int = 3, 
    ):

        def count_common_elements(list1, list2):
            return len(set(list1) & set(list2))

        pred_size = K * len(y.keys())
        correct_size = 0
        for key, value in y.items():
            pred = y_pred[key][:K]
            gt = y[key][:K]

            correct_size = correct_size + count_common_elements(pred, gt)

        return correct_size / pred_size
