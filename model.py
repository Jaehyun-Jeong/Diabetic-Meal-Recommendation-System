from typing import List
import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
import implicit
from implicit.nearest_neighbours import bm25_weight
from implicit.lmf import LogisticMatrixFactorization
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import BaseEstimator

from DataLoader import select_similar_features


FOOD_CATEGORIES = ['과일군', '곡류군', '혼합식품', '어육류군', '우유군', '채소군', '지방군']

# Random Search Result
# {'regularization': 100.0, 'neg_prop': 100, 'learning_rate': 0.001, 'iterations': 250, 'factors': 50}
class LMF(BaseEstimator):

    def __init__(
        self,
        factors: int = 20,
        learning_rate: float = 1.0,
        regularization: float = 1.0,
        iterations: int = 50,
        neg_prop: int = 30,
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

        recommendations = {}
        for user in range(X.shape[0]):
            true_items = X[user].indices
            if len(true_items) == 0:
                continue

            scores = model.model.user_factors[user] @ model.model.item_factors.T
            recommendations[user] = np.argsort(-scores)

        recommendations = pd.DataFrame(recommendations, columns=['patient_id', 'food_id'])

        return recommendations

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
        sim_keys: List[str] = ['Age', 'Gender', 'BMI', 'Body weight ', 'Height '],
        key_x: str = 'patient_id',
        key_y: str = '식품군분류',
        normalize: bool = True,
    ):

        self.bm25_weight = None
        self.base_df = None
        self.K1 = K1
        self.B = B
        self.sim_keys = sim_keys
        self.key_x = key_x
        self.key_y = key_y
        self.normalize = normalize

    def __bm25_weight(
        self,
        train_df: pd.DataFrame,
    ):

        mat = train_df.groupby(
            [self.key_x, self.key_y],
            observed=False,
        ).size().unstack(fill_value=0)
        mat = bm25_weight(
            mat,
            K1=self.K1,
            B=self.B,
        )
        mat = pd.DataFrame.sparse.from_spmatrix(mat)
        mat.index = train_df.groupby(
            [self.key_x, self.key_y],
            observed=False,
        ).size().unstack(fill_value=0).index
        mat.columns = train_df.groupby(
            [self.key_x, self.key_y],
            observed=False,
        ).size().unstack(fill_value=0).columns

        return mat

    def fit(
        self,
        train_df: pd.DataFrame,  # Matrix for BM25
        y=None
    ):

        self.bm25_weight = self.__bm25_weight(train_df, self.key_x, self.key_y)
        self.base_df = select_similar_features(
            train_df,
            keys=self.sim_keys,
        )

        if self.normalize:
            self.base_df = self.normalize_df(self.base_df, self.base_df)

    def predict(
        self,
        X: pd.DataFrame,
    ):

        # Preprocess X
        X = X[self.sim_keys]
        if self.normalize:
            X = self.normalize_df(self.base_df, X)

        # Calculate Similarity
        similarity = cosine_similarity(self.base_df, X)
        score_pred = similarity.T.dot(
            (self.bm25_weight) / np.array([np.abs(similarity).sum(axis=1)]).T
        )

        # Recommend
        recommendations = {}
        for idx, value in enumerate(X.index):
            sorted_recommendations = score_pred[idx].argsort()[::-1]
            recommendations[value] = sorted_recommend

        recommendations = pd.DataFrame(recommendations, columns=['patient_id', 'food_id'])

        return recommendations

    @staticmethod
    def normalize_df(
        base_df: pd.DataFrame,
        converted_df: pd.DataFrame,
    ):

        # Normalize
        mean = base_df.mean()
        std = base_df.std()
        converted_df = (converted_df - mean) / std

        return converted_df

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


class BM25CosSimLMF(BaseEstimator):

    def __init__(
        self,
        K1: float = 3.95,
        B: float = 0.2,
        sim_keys: List[str] = ['Age', 'Gender', 'BMI', 'Body weight ', 'Height '],
        key_x: str = 'patient_id',
        key_y: str = '식품군분류',
        normalize: bool = True,
        factors: int = 20,
        learning_rate: float = 1.0,
        regularization: float = 1.0,
        iterations: int = 50,
        neg_prop: int = 30,
    ):

        self.model_LMF = LMF(
            factors=factors,
            learning_rate=learning_rate,
            regularization=regularization,
            iterations=iterations,
            neg_prop=neg_prop,
        )

        self.model_BM25CosSim = BM25CosSim(
            K1=K1,
            B=B,
            sim_keys=sim_keys,
            key_x=key_x,
            key_y=key_y,
        )

    def fit(
        self,
        train_df: pd.DataFrame,
        y=None,
    ):

        # BM25 + cos similarity
        self.model_BM25CosSim.fit(train_df)

        # Logistic Matrix Factorization
        xui_csr = csr_matrix(self.model_BM25CosSim.bm25_weight)
        self.model_LMF.fit(xui_csr)

    def predict(
        self,
        X: pd.DataFrame,
    ):

        self.model_BM25CosSim.predict(X)
