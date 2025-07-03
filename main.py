import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import scipy
from implicit.nearest_neighbours import bm25_weight
import implicit

from DataLoader import load_data, select_similar_features, split_train_val, create_y_target
from model import BM25CosSim

np.random.seed(42)

patient_df = load_data("./dataset/total_metrics.csv")  # patient information for similarity

FOOD_CATEGORIES = ['과일군', '곡류군', '혼합식품', '어육류군', '우유군', '채소군', '지방군']
GOOD_MEAL_SCORE = 50.0
df = load_data("./dataset/evaluated_meals.csv")
df = df.loc[df['meal_score'] >= GOOD_MEAL_SCORE]

size_val = 10  # use 10 patients to validate
patient_ids = df['patient_id'].unique()
val_ids = np.random.choice(patient_ids, size=size_val, replace=False)
train_df, val_df = split_train_val(
    df=df,
    val_ids=val_ids,
)
patient_train_df, patient_val_df = split_train_val(
    df=patient_df,
    val_ids=val_ids
)
train_df['식품군분류'] = pd.Categorical(train_df['식품군분류'], categories=FOOD_CATEGORIES)

# every column names
# patient_id, meal_time, meal_type, carbs, protein, fat, fiber
# delta_g, g_max, gl, cho_ratio, protein_ratio, fat_ratio
# 식품군분류, good_meal_label, meal_score
# Age, Gender, BMI, Body weight, Height 
keys=['Age', 'Gender', 'BMI', 'Body weight ', 'Height ']
patient_train_df = select_similar_features(
    patient_train_df,
    keys=keys,
)
patient_val_df = select_similar_features(
    patient_val_df,
    keys=keys,
)

model = BM25CosSim()

model.fit(
    train_df=train_df,
    key_x='patient_id',
    key_y='식품군분류',
    sim_df=patient_train_df,
)

recommend_pred = model.predict(patient_val_df)

score = BM25CosSim.recallK(recommend_pred, recommend_target)

print(score)
