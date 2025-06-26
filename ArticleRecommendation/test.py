import pandas as pd
import numpy as np
import random
import os
from scipy.spatial.distance import pdist, squareform
import scipy
from implicit.nearest_neighbours import bm25_weight
import implicit


def seed_everything(
    seed
):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)

view_log = pd.read_csv('dataset/view_log.csv')
article_info = pd.read_csv('dataset/article_info.csv')
sample_submission = pd.read_csv('dataset/sample_submission.csv')

user_article_matrix = view_log.groupby(['userID', 'articleID']).size().unstack(fill_value=0)
user_article_matrix = bm25_weight(user_article_matrix, K1=3.95, B=0.2)
user_article_matrix = pd.DataFrame.sparse.from_spmatrix(user_article_matrix)
user_article_matrix.index = view_log.groupby(['userID', 'articleID']).size().unstack(fill_value=0).index
user_article_matrix.columns = view_log.groupby(['userID', 'articleID']).size().unstack(fill_value=0).columns

# Pairwise cos distance
# return 1d numpy.ndarray
# return with the length of comb(n_user, 2)
user_similarity = pdist(user_article_matrix, metric='cosine')  # (1, n_user * 2)
user_similarity = 1 - squareform(user_similarity)  # (n_user, n_user)

# user/item pair score
# user_article_matrix: bm25 weights
user_predicted_scores = user_similarity.dot(
    (user_article_matrix)/np.array([np.abs(user_similarity).sum(axis=1)]).T
)

# Recommend
recommendations = []
for idx, user in enumerate(user_article_matrix.index):
    sorted_indicies = user_predicted_scores[idx].argsort()[::-1]

    print(sorted_indicies)
    print(user_predicted_scores[idx][385:395])
    raise ValueError("test")

    sorted_recommend = [article for article in user_article_matrix.columns[sorted_indicies]]

    print(sorted_recommend)

    for article in sorted_recommend:
        recommendations.append([user, article])

sorted_recommendations_bm25 = pd.DataFrame(
    recommendations,
    columns=['userID', 'articleID'],
)
