import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

from DataLoader import load_data

GOOD_MEAL_SCORE = 50.0

def preprocess_data():

    df = load_data()
    df = df.loc[df['meal_score'] >= GOOD_MEAL_SCORE]

    features = [
        'meal_type', 'carbs', 'calories', 'protein', 'fat', 'fiber',
        'g0', 'aucg', 'risk_g0', 'Age', 'Gender', 'BMI', 'Body weight ', 'Height '
    ]
    targets = ['delta_g']

    df = df[features + targets]
    print(df.shape)

    raise ValueError("test")

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df = pd.get_dummies(df, columns=categorical_cols, dtype=float)


    # df.to_excel('./output.xlsx', index=False)

    return df

df = preprocess_data()

'''
model = Lasso(alpha=1, tol=1e-7, selection='random')
model.fit(x_train[item], y_train[item])

raw_cv(models, x_train, y_train)

submit(
    "submission/Lasso_multi.csv",
    "./dataset/test_y_v3",
    "./sample_submission.csv",
    models,
    output_size=1,
    ewm=True,
)
'''
