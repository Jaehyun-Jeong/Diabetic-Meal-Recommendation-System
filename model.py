import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import pdist, squareform
import implicit
from implicit.nearest_neighbours import bm25_weight
from implicit.lmf import LogisticMatrixFactorization
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import BaseEstimator



class LMF(BaseEstimator):

    def __init__(
        self,
        factors=20,
        learning_rate=1.0,
        regularization=1.0,
        iterations=50,
        neg_prop=30
    ):

        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.iterations = iterations
        self.neg_prop = neg_prop
        self.model = None

    def fit(self, X, y=None):
        self.model = LogisticMatrixFactorization(
            factors=self.factors,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            iterations=self.iterations,
            neg_prop=self.neg_prop
        )
        self.model.fit(X)  # transpose: shape [items, users]
        return self

    def predict(self, X):
        return self.model.user_factors @ self.model.item_factors.T

# Recall@k 정의
def recall_at_k(model, X, k=3):
    recalls = []
    for user in range(X.shape[0]):
        true_items = X[user].indices
        if len(true_items) == 0:
            continue
        scores = model.model.user_factors[user] @ model.model.item_factors.T
        top_k_items = np.argpartition(-scores, k)[:k]
        hits = np.intersect1d(top_k_items, true_items, assume_unique=True)
        recalls.append(len(hits) / len(true_items))
    return np.mean(recalls)

def recall_scorer(estimator, X_val):
    return recall_at_k(estimator, X_val, k=3)


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
        self.K1 = K1
        self.B = B

    def __bm25_weight(
        self,
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
            K1=self.K1,
            B=self.B,
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
